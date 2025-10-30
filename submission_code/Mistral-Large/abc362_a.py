import sys

input = sys.stdin.read
data = input().split()

R = int(data[0])
G = int(data[1])
B = int(data[2])
C = data[3]

prices = [R, G, B]

if C == "Red":
    prices.remove(R)
elif C == "Green":
    prices.remove(G)
elif C == "Blue":
    prices.remove(B)

print(min(prices))