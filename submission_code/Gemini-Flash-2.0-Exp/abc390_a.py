A = list(map(int, input().split()))

def is_sorted(arr):
  for i in range(len(arr) - 1):
    if arr[i] > arr[i+1]:
      return False
  return True

count = 0
for i in range(len(A) - 1):
  temp = A[:]
  temp[i], temp[i+1] = temp[i+1], temp[i]
  if is_sorted(temp):
    count += 1

if count > 0:
  print("Yes")
else:
  print("No")