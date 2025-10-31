
"""
Práctica 4: Sistema Muscoesqueletico
Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México
Nombre del alumno: Lopez Lopez Santiago
Número de control: 22212260
Correo institucional: L22212260@tectijuana.edu.mx
Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
# Instalar librerias en consola
#!pip install slycot

#!pip install control
import control as ctrl

# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import math as m 
import matplotlib.pyplot as plt 
from scipy import signal
import pandas as pd


# Vector de tiempo
# Vector de tiempo
x0, t0, tend, dt, w, h = 0, 0, 10, 1E-3, 10, 5
N = round((tend - t0)/dt) + 1
t = np.linspace(t0, tend, N)

# Señal escalón 
u = np.zeros(N)
start_index = int(1/dt)  
end_index = int(2/dt)   
u[start_index:end_index] = 1


def musculo(Cp,Cs,R,A):
    num = [Cs*R,1-A]
    den = [R*(Cp+Cs),1]
    sys = ctrl.tf(num,den)
    return sys


#Funcion de transferencia Control
Cp,Cs,R,A = 100e-6,10e-6,100,0.25
syscontrol = musculo(Cp,Cs,R,A)
print (f'Funcion de transferencia de control: {syscontrol}')

#Funcion de transferencia Caso
Cp,Cs,R,A = 100e-6,10e-6,10e3,0.25
syscaso = musculo(Cp,Cs,R,A)
print (f'Funcion de transferencia de caso: {syscaso}')

#Funcion de transferencia F(t)
a,p,pulso,delay = 1,10,10,1
sysft = musculo(a,p,pulso,delay)
print (f'Funcion de transferencia de caso: {sysft}')


#Respuesta en lazo abierto 
 
_,Fs1 = ctrl.forced_response(syscontrol,t,u,x0)
_,Fs2 = ctrl.forced_response(syscaso,t,u,x0)

fg1 = plt.figure()
plt.plot(t,u,'-',linewidth=1,color=[0.23,0.67,0.20],label= 'F(t)')
plt.plot(t,Fs1,'-',linewidth=1,color=[0.93,0.64,0.35],label= 'F(t): control')
plt.plot(t,Fs2,'-',linewidth=1,color=[ 0.90,0.15,0.15],label= 'F(t): caso')
plt.grid(False) #para que salga cuadricula
plt.xlim(0,10);plt.xticks(np.arange(0,11,1))
plt.ylim(-0.1,1.1);plt.yticks(np.arange(-0.1,1.1,0.2))
plt.xlabel('V(t) [V]')
plt.ylabel('t [s]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=3)
plt.show()
fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('sistema muscoesqueletico pyhton.png', dpi=600,bbox_inches='tight')
fg1.savefig('sistema muscoesqueletico pyhton.pdf')

def controlador(kP,kI):
    Cr = 1E-6
    Re = 1/(kI*Cr)
    Rr = kP*Re
    numPI = [Rr*Cr,1]
    denPI = [Re*Cr,0]
    controlPI = ctrl.tf(numPI,denPI)
    return controlPI

PI= controlador(0.21677,3675.5791)
X= ctrl.series(PI,syscaso)
sysPID= ctrl.feedback(X,1,sign=-1)

#Respuesta en lazo cerrado 

_,Fs3 = ctrl.forced_response(sysPID,t,Fs1,x0)

fg2 = plt.figure()
plt.plot(t,u,'-',linewidth=1,color=[0.23,0.67,0.20],label= 'F(t)')
plt.plot(t,Fs1,'-',linewidth=1,color=[0.93,0.64,0.35],label= 'F(t): control')
plt.plot(t,Fs2,'-',linewidth=1,color=[ 0.90,0.15,0.15],label= 'F(t): caso')
plt.plot(t,Fs3,'--',linewidth=1,color=[ 0.90,0.15,0.15],label= 'F(t): Tratamiento')
plt.grid(False) #para que salga cuadricula
plt.xlim(0,10);plt.xticks(np.arange(0,11,1))
plt.ylim(-0.1,1.1);plt.yticks(np.arange(-0.1,1.1,0.2))
plt.xlabel('Ve(t) [V]')
plt.ylabel('t [s]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=3)
plt.show()
fg2.set_size_inches(w,h)
fg2.tight_layout()
fg2.savefig('sistema muscoesqueletico pyhton CONTROLADORPI.png', dpi=600,bbox_inches='tight')
fg2.savefig('sistema muscoesqueletico pyhton CONTROLADORPI.pdf')

