# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
L = int(data[2])

main_dishes = list(map(int, data[3:3+N]))
side_dishes = list(map(int, data[3+N:3+N+M]))
forbidden_pairs = [(int(data[3+N+M+2*i])-1, int(data[3+N+M+2*i+1])-1) for i in range(L)]

# Create a list of all possible set meals and their prices
set_meals = [(main_dishes[i], side_dishes[j], main_dishes[i] + side_dishes[j]) for i in range(N) for j in range(M)]

# Remove the forbidden set meals
set_meals = [meal for meal in set_meals if (meal[0]-1, meal[1]-1) not in forbidden_pairs]

# Find the most expensive set meal
max_price = max(set_meals, key=lambda x: x[2])[2]

print(max_price)