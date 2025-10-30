# YOUR CODE HERE
import math

N = int(input())
good_integers = set()

a = 1
while True:
    power_of_2 = 1 << a  # 2^a
    if power_of_2 > N:
        break
    
    # Find maximum b such that 2^a * b^2 <= N
    max_b_squared = N // power_of_2
    max_b = int(math.sqrt(max_b_squared))
    
    # Add all good integers for this value of a
    for b in range(1, max_b + 1):
        good_integers.add(power_of_2 * b * b)
    
    a += 1

print(len(good_integers))