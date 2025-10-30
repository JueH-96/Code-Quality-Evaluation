n = int(input())
arr = list(map(int, input().split()))
if all(x == arr[0] for x in arr):
    print("Yes")
else:
    print("No")