import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
H = [0] * (N+1)
 for i in range(1, N+1):
     H[i] = int(data[index])
     index += 1

T = [0] * (N+1)
T[1] = H[1] + 1

 for i in range(2, N+1):
     if H[i] <= H[i-1]:
         T[i] = H[i] + T[i-1]
     else:
         T[i] = 2 * H[i] + T[i-1] - H[i-1]

 for i in range(1, N+1):
     if i > 1:
         print(' ', end='')
     print(T[i], end='')
 print()