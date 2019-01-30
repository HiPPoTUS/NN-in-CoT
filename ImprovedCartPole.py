import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import scipy.integrate as inte

def cartpole(x, t, M, m, L, g, u):
	x1, x2, x3, x4 = x
	sin = np.sin(x3)
	cos = np.cos(x3)
	gamma = M+m-3/4*m*cos**2
	dydt = [x2,
			(x4**2*m*L*sin-3/4*m*g*sin*cos + u)/gamma,
			x4,
			(3*g/4/L*(M+m)*sin-3/4*m*x4*sin*cos-3/4/L*u*cos)/gamma]
	return dydt

class ImprovedCartPoleEnv(gym.Env):
	"""docstring for ImprovedCartPoleEnv"""
	metadata = {
		'render.modes' : ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}
	def __init__(self):
		self.gravity = 9.81
		self.masscart = 2.0
		self.masspole = 1.0
		self.length = 0.5
		self.total_mass = (self.masspole + self.masscart)
		self.th_angle = np.pi/2
		self.th_position = 10.0

		self.obs_high = np.array([
			self.th_position,
			np.finfo(np.float32).max,
			self.th_angle,
			np.finfo(np.float32).max])
		self.act_high = 20.0
		self.observation_space = spaces.Box(-self.obs_high, self.obs_high, dtype=np.float32)
		self.action_space = spaces.Box(-self.act_high, self.act_high, shape=(1,), dtype=np.float32)

		self.seed()
		self.viewer = None
		self.state = None
		self.steps_beyond_done = None

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		state = inte.odeint(cartpole, np.array(self.state), np.array([0,0.02]), args=(self.masscart, self.masspole, self.length, self.gravity, action)) 
		x, x_dot, theta, theta_dot = state[1]
		self.state = (x, x_dot, theta, theta_dot)
		done =  x < -self.th_position \
				or x > self.th_position \
				or theta < -self.th_angle \
				or theta > self.th_angle
		done = bool(done)

		if not done:
			tm = ((1-abs(theta/self.th_angle))**2 + (1-abs(x/self.th_position))**2 + (1-abs(action/self.act_high))**2)/3
			reward = tm[0]
		elif self.steps_beyond_done is None:
			# Pole just fell!
			self.steps_beyond_done = 0
			reward = 1.0
		else:
			if self.steps_beyond_done == None:
				logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			self.steps_beyond_done += 1
			reward = 0.0
		return np.array(self.state), reward, done, {}

	def reset(self):
		self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
		self.steps_beyond_done = 0
		return np.array(self.state)

	def render(self, mode='human'):
		screen_width = 600
		screen_height = 400

		world_width = self.th_position*2
		scale = screen_width/world_width
		carty = 100 # TOP OF CART
		polewidth = 10.0
		polelen = scale * (2 * self.length)
		cartwidth = 50.0
		cartheight = 30.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
			axleoffset =cartheight/4.0
			cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.carttrans = rendering.Transform()
			cart.add_attr(self.carttrans)
			self.viewer.add_geom(cart)
			l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
			pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			pole.set_color(.8,.6,.4)
			self.poletrans = rendering.Transform(translation=(0, axleoffset))
			pole.add_attr(self.poletrans)
			pole.add_attr(self.carttrans)
			self.viewer.add_geom(pole)
			self.axle = rendering.make_circle(polewidth/2)
			self.axle.add_attr(self.poletrans)
			self.axle.add_attr(self.carttrans)
			self.axle.set_color(.5,.5,.8)
			self.viewer.add_geom(self.axle)
			self.track = rendering.Line((0,carty), (screen_width,carty))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)

			self._pole_geom = pole

		if self.state is None: return None

		# Edit the pole polygon vertex
		pole = self._pole_geom
		l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
		pole.v = [(l,b), (l,t), (r,t), (r,b)]
		
		x = self.state
		cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
		self.carttrans.set_translation(cartx, carty)
		self.poletrans.set_rotation(-x[2])
		
		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None