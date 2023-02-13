import matplotlib
matplotlib.use('TkAgg') # 'tkAgg' if Qt not present 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np
import random
import pandas as pd
  
class Pendulum:
    def __init__(self, theta1, omega1, theta2, omega2, dt):
        self.theta1 = theta1
        self.theta2 = theta2
        
        self.m1 = 1.0
        self.m2 = 1.0
          
        self.dt = dt
          
        self.g = 9.81
        self.length = 1.0
        
        # Calculates the Hamiltonian conjugate momenta p1 and p2 for each of the angles
        self.p1 = (self.m1 + self.m2) * (self.length ** 2) * omega1 + self.m2 * (self.length ** 2) * omega2 * np.cos(self.theta1 - self.theta2)
        self.p2 = self.m2 * (self.length ** 2) * omega2 + self.m2 * (self.length ** 2) * omega1 * np.cos(self.theta1 - self.theta2)
          
        self.Hamiltonian = round(0.5 * (self.m1 + self.m2) * ((self.length * omega1) ** 2) + 0.5 * self.m2 * ((self.length * omega2) ** 2) + self.m2 * (self.length ** 2) * omega1 * omega2 * np.cos(self.theta1 - self.theta2) + self.g * self.length * (self.m1 + self.m2) * (1 - np.cos(self.theta1)) + self.g * self.length * self.m2 * (1 - np.cos(self.theta2)), 2)
        
        # List containing the position history of the pendulum masses
        self.trajectory = [self.polar_to_cartesian()]
    
    # Returns the Cartesian coordinates of both masses, and the position of the origin
    def polar_to_cartesian(self):
        x1 =  self.length * np.sin(self.theta1)        
        y1 = -self.length * np.cos(self.theta1)
          
        x2 = x1 + self.length * np.sin(self.theta2)
        y2 = y1 - self.length * np.cos(self.theta2)
        
        return np.array([[0.0, 0.0], [x1, y1], [x2, y2]])
      
    # Uses Hamilton's equations to evolve the system
    # by time step dt using a Riemann sum approach to integration
    def evolve(self):
        theta1 = self.theta1
        theta2 = self.theta2
        p1 = self.p1
        p2 = self.p2
        g = self.g
        l = self.length
        
        # Calculates quantities of relevance in the eoms
        expr1 = np.cos(theta1 - theta2)
        expr2 = np.sin(theta1 - theta2)
        expr3 = (1 + expr2**2)
        expr4 = p1 * p2 * expr2 / expr3
        expr5 = (p1**2 + 2 * p2**2 - p1 * p2 * expr1) \
        * np.sin(2 * (theta1 - theta2)) / 2 / expr3**2
        expr6 = expr4 - expr5
        
        # Updates the variables using the eoms
        self.theta1 += self.dt * (p1 - p2 * expr1) / expr3
        self.theta2 += self.dt * (2 * p2 - p1 * expr1) / expr3
        self.p1 += self.dt * (-2 * g * l * np.sin(theta1) - expr6)
        self.p2 += self.dt * (    -g * l * np.sin(theta2) + expr6)
         
        new_position = self.polar_to_cartesian()
        self.trajectory.append(new_position)
        # self.Hamiltonian = round(0.5 * (self.m1 + self.m2) * ((self.length * omega1) ** 2) + 0.5 * self.m2 * ((self.length * omega2) ** 2) + self.m2 * (self.length ** 2) * omega1 * omega2 * np.cos(self.theta1 - self.theta2) + self.g * self.length * (self.m1 + self.m2) * (1 - np.cos(self.theta1)) + self.g * self.length * self.m2 * (1 - np.cos(self.theta2)), 2)
        return new_position
    
    # Generates a dictionary containing n data points containing
    # the energy and positions at each time step
    def generate_n_data(self, n):
        output_list = [[],[],[],[],[]]
        for i in range(n):
            current_variables = [self.Hamiltonian, self.theta1, self.theta2, self.p1, self.p2]
            for j in range(5):
                output_list[j].append(current_variables[j])
            self.evolve()
        output_dict = {
            "Hamiltonian": output_list[0],
            "Theta1": output_list[1],
            "Theta2": output_list[2],
            "Velocity1": output_list[3],
            "Velocity2": output_list[4]}    
        return output_dict
 
# Is used to animate double pendulum trajectories over time
class Animator:
    def __init__(self, pendulum, draw_trace=False):
        self.pendulum = pendulum
        self.draw_trace = draw_trace
        self.time = 0.0
  
        # set up the figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_xlim(-2.5, 2.5)
  
        # prepare a text window for the timer
        self.time_text = self.ax.text(0.05, 0.95, '', 
            horizontalalignment='left', 
            verticalalignment='top', 
            transform=self.ax.transAxes)
            
        # prepare a text window to display the Hamiltonian
        self.Ham_text = self.ax.text(0.05, 0.85, 'Hamiltonian = ' + str(pendulum.Hamiltonian), horizontalalignment='left', 
            verticalalignment='top', 
            transform=self.ax.transAxes)
        
  
        # initialize by plotting the last position of the trajectory
        self.line, = self.ax.plot(
            self.pendulum.trajectory[-1][:, 0], 
            self.pendulum.trajectory[-1][:, 1], 
            marker='o')
          
        # trace the whole trajectory of the second pendulum mass
        if self.draw_trace:
            self.trace, = self.ax.plot(
                [a[2, 0] for a in self.pendulum.trajectory],
                [a[2, 1] for a in self.pendulum.trajectory])
     
    def advance_time_step(self):
        while True:
            self.time += self.pendulum.dt
            yield self.pendulum.evolve()
             
    def update(self, data):
        self.time_text.set_text('Elapsed time: {:6.2f} s'.format(self.time))
         
        self.line.set_ydata(data[:, 1])
        self.line.set_xdata(data[:, 0])
         
        if self.draw_trace:
            self.trace.set_xdata([a[2, 0] for a in self.pendulum.trajectory])
            self.trace.set_ydata([a[2, 1] for a in self.pendulum.trajectory])
        return self.line,
     
    def animate(self):
        self.animation = animation.FuncAnimation(self.fig, self.update,
                         self.advance_time_step, interval=25, blit=False)

# Creates an animation of a pendulum with random starting conditions
def animate_random():
    pendulum = Pendulum(theta1=random.uniform(-np.pi,np.pi), omega1=random.uniform(-10,10), theta2=random.uniform(-np.pi,np.pi), omega2=random.uniform(-10,10), dt=0.01)
    animator = Animator(pendulum=pendulum, draw_trace=True)
    animator.animate()
    plt.show()

# Creates a csv file containing n data points (energy + coords)
# for m pendula and returns a dictionary of these points
def n_data_m_pendula(n, m):
    output_list = [[],[],[],[],[]]
    for num in range(m):
        pendulum = Pendulum(theta1=random.uniform(-np.pi,np.pi), omega1=random.uniform(-10,10), theta2=random.uniform(-np.pi,np.pi), omega2=random.uniform(-10,10), dt=0.01)
        for i in range(n):
            current_variables = [pendulum.Hamiltonian, pendulum.theta1, pendulum.theta2, pendulum.p1, pendulum.p2]
            for j in range(5):
                output_list[j].append(current_variables[j])
            pendulum.evolve()
        for k in range(5):
            output_list[k].append("-----")
    output_dict = {
        "Hamiltonian": output_list[0],
        "Theta1": output_list[1],
        "Theta2": output_list[2],
        "Velocity1": output_list[3],
        "Velocity2": output_list[4]}   
    data_frame = pd.DataFrame(output_dict)
    print(data_frame)
    with open('%d_data_%d_pendula.txt' % (n, m), 'w') as csv_file:
        data_frame.to_csv(path_or_buf=csv_file)
    return output_dict
    
def new_n_data_m_pendula(n,m):
    output_list = []
    for num in range(m):
        current_data = []
        pendulum = Pendulum(theta1=random.uniform(-np.pi,np.pi), omega1=random.uniform(-10,10), theta2=random.uniform(-np.pi,np.pi), omega2=random.uniform(-10,10), dt=0.01)
        for i in range(n):
            current_variables = [pendulum.theta1, pendulum.theta2, pendulum.p1, pendulum.p2]
            current_data = [*current_data, *current_variables]
        output_list.append((current_data, [pendulum.Hamiltonian]))
    data_frame = pd.DataFrame(output_list)
    print(data_frame)
    with open('%d_data_%d_pendula.txt' % (n, m), 'w') as csv_file:
        data_frame.to_csv(path_or_buf=csv_file)
    return output_list
    
new_n_data_m_pendula(10, 100)