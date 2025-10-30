# YOUR CODE HERE
import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index+N]))
index += N

S = []
sum_s = 0

 for i in range(N):
     a_i = A[i]
     if len(S) > 0 and S[-1] < 0:
         # Choose to delete the last element
         sum_s -= S.pop()
     else:
         # Choose to append A_i
         S.append(a_i)
         sum_s += a_i

 print(sum_s)