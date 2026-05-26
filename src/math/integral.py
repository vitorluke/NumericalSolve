import numpy as np

def trapezoid_method(f, l, r, n):
    delta_x = (r-l)/n
    prev = f(l)
    sum = prev
    for i in range(1, n-1):
        height = f(l + i * delta_x)
        #print(height, l + i * delta_x)
        sum += prev + height
        prev = height
    sum += f(r)
    return sum * delta_x / 2




if __name__ == "__main__":
    def f(x):
        return x ** 2;
    l = 10
    r = 100
    expected_integral = (r**3 - l **3)/3
    S = trapezoid_method(f,l,r,100000)
    print(f"Valor verdadeiro: {expected_integral} \n Valor numérico: {S} \n Erro: {1-S/expected_integral:0.4%}")
        