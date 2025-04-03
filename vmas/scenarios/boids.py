import math
import numpy as np
from vmas.simulator import rendering
from vmas.simulator.core import Agent, Landmark, World, Shape, Sensor
from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.rendering import Geom
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils, Color
import torch
from torch import Tensor
from typing import List


class Scenario(BaseScenario):
  #####################################################################
  ###                    make_world function                        ###
  #####################################################################
  def make_world(self, batch_dim: int, device: torch.device, **kwargs):
    self.batch_dim = batch_dim
    self.device=device
    self.world_size_x = kwargs.pop("world_size_x", 3)
    self.world_size_y = kwargs.pop("world_size_y", 3)
    self.plot_grid = True

    ##############################
    ###### INFO FROM KWARGS ######
    ##############################

    ''' Entity spawning '''
    self.world_spawning_x = kwargs.pop("world_spawning_x", self.world_size_x * 0.9)
    self.world_spawning_y = kwargs.pop("world_spawning_y", self.world_size_y * 0.5)
    self.min_distance_between_entities = (kwargs.pop("agent_radius", 0.1) * 2 + 0.05)

    ''' Agent entities '''
    self.n_agents = kwargs.pop("n_agents", 10)
    self.n_teams = kwargs.pop("teams", 2)

    ''' Goal entities '''
    self.n_goals = kwargs.pop("n_goals", 7)
    self.goal_color = kwargs.pop("goal_colour", Color.YELLOW)

    self.flat_goal_reward = kwargs.pop("flat_goal_reward", 100)
    self.flat_goal_reward = torch.full((self.batch_dim,), self.flat_goal_reward, device=self.device)

    self.goal_range = kwargs.pop("goal_range", 0.4)
    self.goal_threshold = kwargs.pop("goal_threshold", 2)
    self.goal_respawn = kwargs.pop("goal_respawn", True)

    ''' Reward Info '''
    self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
    self.wall_collision_penalty = kwargs.pop("wall_collision_penalty", -1)
    self.shared_rew = kwargs.pop("shared_rew", False)
    self.min_collision_distance = (0.005)

    ''' Warn if not all kwargs have been consumed '''
    ScenarioUtils.check_kwargs_consumed(kwargs)

    ####################################
    ######### MAKING THE WORLD #########
    ####################################

    world = World(
      batch_dim=batch_dim,  # Number of environments
      device=device,  # Use your hardware (GPU/CPU)
      substeps=5,  # Substeps for simulation accuracy
      collision_force=500,  # Collision force for agent interaction
      dt=0.1,  # Simulation timestep
      drag=0.1,  # Optional drag for agent movement
      linear_friction=0.05,  # Optional friction
      angular_friction=0.02,  # Optional angular friction
      x_semidim=self.world_size_x, # bounds of the world
      y_semidim=self.world_size_y, # bounds of the world
    )

    known_colors = [
          Color.GREEN, # Team 1
          Color.RED,    # Team 2
          Color.YELLOW # Rewards
    ]
    self.goals = []

    ''' Adding agents '''
    self.teams = {}
    for team in range(self.n_teams):
      self.teams[team] = []
      for agent_num in range(int(self.n_agents)):

        sensors = [SenseSphere(world)]
        agent = Agent(
          name=f"team_{team}_agent_{agent_num}",
          collide=True,
          rotatable=True,
          color=known_colors[team],
          render_action=True,
          sensors=sensors,
          shape=Triangle(),
          u_range=[1],  # Ranges for actions
          u_multiplier=[1],  # Action multipliers
          dynamics=BoidDynamics(world=world, team=team)
        )

        agent.pos_rew = torch.zeros(
          batch_dim, device=device
        )  # Tensor that will hold the position reward fo the agent
        agent.agent_collision_rew = (
          agent.pos_rew.clone()
        )  # Tensor that will hold the collision reward fo the agent

        self.teams[team].append(agent)
        world.add_agent(agent)

    ''' Adding Goals  '''
    for i in range(self.n_goals):
      goal = Landmark(
        name=f"goal_{i}",
        collide=True,
        movable = False,
        color=known_colors[2],
      )
      goal.range = self.goal_range
      goal.threshold = self.goal_threshold
      self.goals.append(goal)
      world.add_landmark(goal)

    return world

  #####################################################
  ############## reset_world_at function ##############
  #####################################################
  def reset_world_at(self, env_index: int = None):
    # Spawn friendlies
    ScenarioUtils.spawn_entities_randomly(
      self.teams[0],
      self.world,
      env_index, # Pass the env_index so we only reset what needs resetting
      self.min_distance_between_entities,
      x_bounds=(-self.world_size_x, self.world_size_x),
      y_bounds=(-self.world_size_y, -self.world_size_y),
    )
    # Spawn Enemies
    ScenarioUtils.spawn_entities_randomly(
      self.teams[1],
      self.world,
      env_index, # Pass the env_index so we only reset what needs resetting
      self.min_distance_between_entities,
      x_bounds=(-self.world_size_x, self.world_size_x),
      y_bounds=(self.world_size_y, self.world_size_y),
    )
    # Spawn Goals
    ScenarioUtils.spawn_entities_randomly(
      self.goals,  # List of entities to spawn
      self.world,
      env_index, # Pass the env_index so we only reset what needs resetting
      self.min_distance_between_entities,
      x_bounds=(-self.world_spawning_x, self.world_spawning_x),
      y_bounds=(-self.world_spawning_y, self.world_spawning_y),
    )

    for agent in self.world.agents:
      agent.dynamics.reset(0)

  ##################################################
  ############## observation function ##############
  ##################################################
  def observation(self, agent: Agent):
    # Flatten and concatenate sensor data into a single tensor
    obs = {
        "obs": torch.cat(
            [agent.state.pos - goal.state.pos for goal in self.goals] +
            [agent.state.pos - teammate.state.pos for teammate in self.teams[agent.dynamics.team]] +
            [agent.state.pos - enemy.state.pos for enemy in self.teams[agent.dynamics.team ^ agent.dynamics.team]],
            dim=-1
        ),
        "pos": agent.state.pos,
        "vel": agent.state.vel,
    }
    return obs

  #############################################
  ############## reward function ##############
  #############################################
  def reward(self, agent: Agent):
    is_first = agent == self.world.agents[0]
    is_last = agent == self.world.agents[-1]

    ''' Taken from the discovery.py scenario '''
    goal_reward = torch.zeros((self.batch_dim,), device=self.device)
    if is_first:
      ''' negative reward for time passing - don't think it's relevant for BOIDS '''
      # self.time_rew = torch.full(
      #   (self.world.batch_dim,),
      #   self.time_penalty,
      #   device=self.world.device,
      # )

      ''' updating tensor of all agent positions - shape [board_dimensions, num_agents] '''
      self.agents_pos = torch.stack(
        [a.state.pos for a in self.world.agents], dim=1
      )
      ''' updating tensor of all goal positions '''
      self.goals_pos = torch.stack([g.state.pos for g in self.goals], dim=1)

      ''' getting tensor with distances between reward positions and agent positions '''
      self.agents_goals_dists = torch.cdist(self.agents_pos, self.goals_pos)

      self.agents_per_goal = torch.sum(
        (self.agents_goals_dists < self.goal_range).type(torch.int),
        dim=1,
      )

      self.covered_goals = self.agents_per_goal >= self.goal_threshold

      ''' Flat reward for each goal captured by the team '''
      goal_reward = torch.zeros((self.batch_dim,), device=self.device)
      # print(self.covered_goals)
      # print(self.covered_goals.shape())
      for goal_covered in self.covered_goals[0]:
        if goal_covered:
            goal_reward += self.flat_goal_reward  # Add flat reward for each goal covered

    if is_last:
      if self.goal_respawn:
        occupied_positions_agents = [self.agents_pos]
        for i, goal in enumerate(self.goals):
          occupied_positions_goals = [
            o.state.pos.unsqueeze(1)
            for o in self.goals
            if o is not goal
          ]
          occupied_positions = torch.cat(
            occupied_positions_agents + occupied_positions_goals,
            dim=1,
          )
          pos = ScenarioUtils.find_random_pos_for_entity(
            occupied_positions,
            env_index=None,
            world=self.world,
            min_dist_between_entities=self.min_distance_between_entities,
            x_bounds=(-self.world.x_semidim, self.world.x_semidim),
            y_bounds=(-self.world.y_semidim, self.world.y_semidim),
          )

          goal.state.pos[self.covered_goals[:, i]] = pos[
            self.covered_goals[:, i]
          ].squeeze(1)
      else:
        self.all_time_covered_goals += self.covered_goals
        for i, goal in enumerate(self.goals):
          goal.state.pos[self.covered_goals[:, i]] = self.get_outside_pos(
            None
          )[self.covered_goals[:, i]]

    ''' Negative reward for touching boundaries (walls) '''
    coll_pen = torch.where(abs(agent.state.pos[0][0]) == self.world_size_x or abs(agent.state.pos[0][1]) == self.world_size_y, -1, 0)
    
    ''' Combining goal reward and collision penalty '''
    reward = goal_reward + coll_pen

    ''' Return the total reward (tensor of shape [self.world.batch_dim,]) '''
    return reward
  
  # def reward(self, agent: Agent):
  #   reward = 0.0
  #   # negative reward for touching boundaries
  #   pos_x = agent.state.pos[0][0]
  #   pos_y = agent.state.pos[0][1]
  #   if abs(pos_x) == self.world_size_x or abs(pos_y) == self.world_size_y:
  #     reward -=1

  #   #return torch.tensor([reward], device=self.device)
  #   return torch.full((self.batch_dim,), reward, device=self.device)


  #############################################
  ############## Extra_render ################
  #############################################
  def extra_render(self, env_index: int = 0) -> "List[Geom]":
    from vmas.simulator import rendering

    geoms: List[Geom] = []

    # Goal covering ranges
    for goal in self.goals:
      range_circle = rendering.make_circle(goal.range, filled=False)
      xform = rendering.Transform()
      xform.set_translation(*goal.state.pos[env_index])
      range_circle.add_attr(xform)
      range_circle.set_color(*self.goal_color.value)
      geoms.append(range_circle)

    return geoms



class BoidDynamics(Dynamics):
    def __init__(self, world, constant_speed=0.5, max_steering_rate=1*math.pi, team=0):
        super().__init__()
        self.constant_speed = constant_speed
        self.max_steering_rate = max_steering_rate  # max radians per second
        self.world = world
        self.team = team

    @property
    def needed_action_size(self):
        return 1  # Steering input only

    def reset(self, env_index: int):
        if self.team == 0:
            self._agent.state.rot += 0.5*math.pi
        elif self.team == 1:
            self._agent.state.rot -= 0.5*math.pi
        self._agent.state.rot = (self._agent.state.rot + math.pi) % (2 * math.pi) - math.pi
        self._agent.state.vel = self.constant_speed * torch.cat(
            [
                torch.cos(self._agent.state.rot),
                torch.sin(self._agent.state.rot)
            ],
            dim=1
        )
        self._agent.state.force = torch.zeros_like(self._agent.state.force)

    def zero_grad(self):
        pass

    def clone(self):
        return BoidDynamics(self.world, self.constant_speed, self.max_steering_rate)
    
    def process_action(self):
        dt = self.world.dt
        #print(dt, self.constant_speed)
        steering_rate = self._agent.action.u[:, 0].clamp(-1, 1) * self.max_steering_rate
        # Update orientation
        self._agent.state.rot -= steering_rate.unsqueeze(1) * dt
        self._agent.state.rot = (self._agent.state.rot + math.pi) % (2 * math.pi) - math.pi

        # Update velocity based on orientation
        self._agent.state.vel = self.constant_speed * torch.cat(
            [
                torch.cos(self._agent.state.rot),
                torch.sin(self._agent.state.rot)
            ],
            dim=1
        )

        # Set force to zero to avoid external forces
        #self._agent.state.force = torch.zeros_like(self._agent.state.force)



class Triangle(Shape):
  def __init__(self, base: float = 0.1, height: float = 0.15):
    assert base > 0, f"Base must be > 0, got {base}"
    assert height > 0, f"Height must be > 0, got {height}"
    self.base = base
    self.height = height

  def is_point_in_triangle(self, x: float, y: float) -> bool:
    # Check if point is within the bounds of the triangle's base and height
    if y < -self.height/2 or y > self.height/2:  # Out of vertical bounds
      return False
    if x < -self.base / 2 or x > self.base / 2:  # Out of horizontal bounds
      return False

    y += (self.height/2)
    slope = self.height / (self.base / 2)
    if x < 0:  # x is to the left of the center and is negative
      max_y = slope * ((self.base/2) + x)
      return y <= max_y
    elif x > 0:  # x is to the right of the center and is positive
      max_y = self.height - slope * (x)
      return y <= max_y
    else:
      return True

  def closest_point_on_segment(self, P, A, B):
    """
    Calculate the closest point from point P to the line segment AB.

    Parameters:
      P (np.array): The point from which we are projecting.
      A (np.array): The start point of the segment.
      B (np.array): The end point of the segment.

    Returns:
      np.array: The closest point on the segment.
    """
    # Vector AB
    AB = B - A
    # Vector AP
    AP = P - A
    # Projection scalar t
    t = np.dot(AP, AB) / np.dot(AB, AB)

    # If the projection is within the segment, return the projected point
    if 0 <= t <= 1:
      closest_point = A + t * AB
    # If the projection falls before A, return A
    elif t < 0:
      closest_point = A
    # If the projection falls after B, return B
    else:
      closest_point = B

    return closest_point

  def shorten_vector_to_triangle(self, vector):
    """
    Shortens a vector so it stops at the perimeter of a triangle centered at the origin.

    Parameters:
        vector (np.array): The 2D vector originating from the origin.

    Returns:
        tuple: The shortened vector that ends at the triangle's perimeter as (x, y).
    """
    # Triangle vertices (centered at origin)
    A = np.array([-self.base / 2, -self.height / 2])  # Left base
    B = np.array([self.base / 2, -self.height / 2])   # Right base
    C = np.array([0, self.height / 2])                # Top

    # Determine which side of the triangle the vector is pointing to
    if vector[0] > 0:
      # Vector is pointing to the right, closest edge is BC
      closest_point = self.closest_point_on_segment(vector, B, C)
    else:
      # Vector is pointing to the left, closest edge is AC
      closest_point = self.closest_point_on_segment(vector, A, C)

    # Return as tuple with normal float type (not np.float64)
    return tuple(float(x) for x in closest_point)

  def get_delta_from_anchor(self, anchor):
    x, y = anchor
    x_box = x * self.base / 2
    y_box = y * self.height / 2

    if self.is_point_in_triangle(x_box, y_box):
      return float(x_box), float(y_box)  # Convert to plain float
    else:
      return self.shorten_vector_to_triangle([x_box, y_box])

  def moment_of_inertia(self, mass: float):
    return (1 / 18) * mass * (self.base**2 + self.height**2)

  def circumscribed_radius(self):
    return math.sqrt(self.base**2 + self.height**2) / 2

  def get_geometry(self) -> "Geom":
    # Vertices of the triangle (centered at the origin)
    A = (-self.height * 0.5, self.base * 0.5)  # Left base
    B = (-self.height * 0.5, -self.base * 0.5)   # Right base
    C = (self.height * 0.5, 0)                # Top

    return rendering.make_polygon([A, B, C])



class SenseSphere(Sensor):
  def __init__(self, world, range=1.0):
    super().__init__(world)
    self.range = range
    self._last_measurement = None

  def measure(self):
    agent_pos = self.agent.state.pos  # Current agent's position
    agent_vel = self.agent.state.vel  # Current agent's velocity

    observations = []
    for other_agent in self._world.agents:
      if other_agent is self.agent:
        continue  # Skip self

      # Compute relative position and velocity
      rel_pos = other_agent.state.pos - agent_pos
      rel_vel = other_agent.state.vel - agent_vel
      distance = torch.norm(rel_pos, dim=-1)
      # Only include agents within range
      # if distance <= self.range:
      #   observations.append(torch.concat([rel_pos, rel_vel], axis=1))
      # else:
      #   observations.append(torch.zeros((1,4)))
      observations.append(torch.concat([rel_pos, rel_vel], axis=1))
    return torch.stack(observations)


  def render(self, env_index: int = 0) -> "List[Geom]":
    # if not self._render:
    #     return []

    geoms: List[rendering.Geom] = []

    # Render the range of the SenseSphere as a circle around each agent
    circle = rendering.make_circle(radius=self.range)  # Create the sensor's circle based on range
    circle.set_color(0, 0, 1, alpha=0.05)  # Set the color to blue with transparency
    xform = rendering.Transform()
    xform.set_translation(*self.agent.state.pos[env_index])  # Position the circle at the agent's position
    circle.add_attr(xform)

    geoms.append(circle)
    return geoms

  def to(self, device: torch.device):
    #self.range = self.range.clone().detach().requires_grad_(True)
    self.range = torch.tensor(self.range, device=device)  # Ensure range is a tensor
