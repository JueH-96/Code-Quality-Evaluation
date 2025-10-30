# YOUR CODE HERE

N = int(input())
sticks = set()

for _ in range(N):
    stick = input()
    stick = min(stick, stick[::-1])
    sticks.add(stick)

print(len(sticks))