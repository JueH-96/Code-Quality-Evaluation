import sys
from collections import deque

def main():
    S = sys.stdin.readline().strip()
    n = len(S)
    distance = [float('inf')] * (n + 1)
    distance[0] = 0
    queue = deque([0])
    buttons = ['00', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    while queue:
        i = queue.popleft()
        if distance[i] == float('inf'):
            continue
        for button in buttons:
            k = len(button)
            j = i + k
            if j > n:
                continue
            if S[i:i+k] == button and distance[j] > distance[i] + 1:
                distance[j] = distance[i] + 1
                queue.append(j)
    
    print(distance[n])

if __name__ == "__main__":
    main()