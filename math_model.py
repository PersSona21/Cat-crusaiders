import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.integrate import odeint



stages = [ 
    {'mass_stage': 50_000, 'mass_fuel': 110_000, 'burn_time': 90, 'f_tract': 3_000_000}, 
    {'mass_stage': 20_000, 'mass_fuel': 30_000, 'burn_time': 65, 'f_tract': 700_000}, 
] 
e = 2.718281828459045
pi = 3.14


MU = 0.028  # молярная масса воздуха, кг/моль 
R_KERBIN = 600000  # радиус Кербина, метры
C = 0.7 #коэф лобового сопротивления
S = 45  # площадь поперечного сечения, м2 
Po0 = 1.225  # плотность воздуха, кг/м3
G = 6.674e-11  # гравитационная постоянная, Н*м2/кг2
M_KERBIN = 5.2915793e22  #масса Кербина в кг
R = 8.314  # универсальная газовая постояннная, дж/(моль*К) 
START_T = 300  # начальная температура, К
P0 = 101_325  # давление, Паcкаль 
g = 9.81  # ускорение свободного падения


def corner(height: float) -> float:
    if height < 80000: 
        return 90 * (1 - height / 80000) 
    return 0 


def fuel_consumption(mass_fuel: float, time: float) -> float: 
    return mass_fuel / time 


mass_rocket = 210000  # кг масса ракеты
temperature = 300  # температура ракеты 


def equation_odint(x: list, t: list, num_stage: int) -> list:


    pos_x, speed_x, pos_y, speed_y = x

    global temperature

    stage = stages[num_stage]
    alpha = np.radians(corner(pos_y)) # угол поворота ракеты
    mass_fuel = stage["mass_fuel"]
    F_traction = stage['f_tract']
    burn_time = stage['burn_time']
    k = fuel_consumption(mass_fuel, burn_time)
    
    if temperature > 50:
        temperature = START_T - 6 * (pos_y // 1000)
    
    p = P0 * e ** ((-g * pos_y * MU) / (R * temperature)) # давление от высоты
    po = (p * MU) / (R * temperature) # плотность от высоты
    v = speed_x ** 2 + speed_y ** 2    
    cur_mass = mass_rocket - k * t # текущая масса
    
    
    F_grav = (G * M_KERBIN * cur_mass) / ((R_KERBIN + pos_y) ** 2) # сила гравитации
    F_res = C * S * v * po / 2 # сила сопротивления
    
    dspeed_x = (F_traction - F_res) * np.cos(alpha) / cur_mass
    dspeed_y = ((F_traction - F_res) * np.sin(alpha) - F_grav) / cur_mass 
    
    return [speed_x, dspeed_x, speed_y, dspeed_y]


start_value = [0, 0, 0, 0] # начальные значения

# первая ступень 
time1 = np.linspace(0, stages[0]["burn_time"]) # Время работы первой ступени 
result1 = odeint(equation_odint, start_value, time1, args=(0,)) # Решение системы для первой ступени 
 
# вторая ступень 
mass_rocket -= (stages[0]['mass_stage'] + stages[0]['mass_fuel']) # Масса ракеты после отсоединения первой ступени 
time2 = np.linspace(0, stages[1]["burn_time"], 100) # Время работы второй ступени 
result2 = odeint(equation_odint, result1[-1, :], time2, args=(1,)) # Решение системы для второй ступени
  
 
# Объединение результатов 
time = np.concatenate([time1, time1[-1] + time2]) 
pos_x = np.concatenate([result1[:, 0], result2[:, 0]]) 
speed_x = np.concatenate([result1[:, 1], result2[:, 1]]) 
pos_y = np.concatenate([result1[:, 2], result2[:, 2]]) 
speed_y = np.concatenate([result1[:, 3], result2[:, 3]])



graphics = dict()
with open("data.json", mode='r', encoding="UTF-8") as f:
    graphics = json.load(f)
    
timeKSP = graphics["time"]
m = len(timeKSP)

x_coords_KSP = graphics["x_coords"][:m]
y_coords_KSP = graphics["y_coords"][:m]
speedX_KSP = graphics["speedX"][:m]
speedY_KSP = graphics["speedY"][:m]


plt.figure(figsize=(12, 12))  

plt.subplot(2, 2, 1)
plt.plot(time, pos_y)
plt.plot(timeKSP, y_coords_KSP, color="red", label="Координата по Y KSP")
plt.xlabel('Время (с)')
plt.ylabel('Координата по Y')


plt.subplot(2, 2, 2)
plt.plot(time, pos_x)
plt.plot(timeKSP, x_coords_KSP, color="red", label="Координата по X KSP")
plt.xlabel('Время (с)')
plt.ylabel('Координата по X (м)')

plt.subplot(2, 2, 3)
plt.plot(time, speed_y)
plt.plot(timeKSP, speedY_KSP, color="red", label="Скорость по Y в KSP")
plt.xlabel('Время')
plt.ylabel('Скорость по Y')


plt.subplot(2, 2, 4)
plt.plot(time, speed_x)
plt.plot(timeKSP, speedX_KSP, color="red", label="Скорость по X KSP")
plt.xlabel('Время (с)')
plt.ylabel('Скорость по X')


plt.show()
