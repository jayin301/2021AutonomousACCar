import math
def curvetosteering(curve):
    if curve>355.3 or curve<67.5:
      steering=0
    else:
      steering=50.353*(curve)**(-0.97)
    return round(steering,2)  
c=84.34 #여기에 측정된 곡률을 대입하게 해서 사용하면될듯....
y=curvetosteering(c)
print(y)
