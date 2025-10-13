# src/05_rl_env.py
# You need to install gymnasium and stable-baselines3: pip install gymnasium stable-baselines3
import gymnasium as gym
from gymnasium import spaces
import carla
import numpy as np
import random
import time

# (This is a simplified environment. A full implementation would be more complex,
# handling various sensors, multiple scenarios, etc.)

class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()
        
        # Action space: [steering, throttle, brake] - continuous values
        self.action_space = spaces.Box(low=np.array([-1, 0, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
        
        # Observation space: RGB image
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.last_image = None
        self.collision_history = []

    def reset(self, seed=None, options=None):
        self.cleanup()
        
        # Spawn vehicle
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Add sensors
        self._setup_sensors()
        
        # Wait for first image
        while self.last_image is None:
            time.sleep(0.01)
            self.world.tick()

        # Reset collision history for the new episode
        self.collision_history = []
        
        info = {} # You can add additional info here if needed
        return self.last_image, info

    def step(self, action):
        steer, throttle, brake = action
        control = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        self.vehicle.apply_control(control)
        
        # Wait for the next frame
        self.world.tick()
        
        # Calculate reward
        done = False
        reward = 0
        
        # Reward for moving forward
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        reward += speed * 0.1

        # Penalty for collision
        if len(self.collision_history) > 0:
            done = True
            reward -= 100
        
        if self.last_image is None:
            # Handle rare case where image is not received
            obs = np.zeros(self.observation_space.shape)
        else:
            obs = self.last_image

        return obs, reward, done, False, {} # obs, reward, terminated, truncated, info

    def _setup_sensors(self):
        # Camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.observation_space.shape[1]))
        camera_bp.set_attribute('image_size_y', str(self.observation_space.shape[0]))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: self._process_image(image))

        # Collision Sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.collision_history.append(event))

    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.last_image = array

    def cleanup(self):
        if self.camera:
            self.camera.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        self.camera = self.collision_sensor = self.vehicle = None
        self.last_image = None

    def close(self):
        self.cleanup()