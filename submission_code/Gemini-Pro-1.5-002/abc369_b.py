# YOUR CODE HERE
n = int(input())
a = []
s = []
for _ in range(n):
    ai, si = input().split()
    a.append(int(ai))
    s.append(si)

left = []
right = []
for i in range(n):
    if s[i] == 'L':
        left.append(a[i])
    else:
        right.append(a[i])

if not left:
    ans = abs(right[-1] - right[0])
    print(ans)
elif not right:
    ans = abs(left[-1] - left[0])
    print(ans)
else:
    ans1 = abs(left[0] - right[0]) + abs(left[-1] - right[-1]) + abs(right[-1] - left[0])
    ans2 = abs(left[0] - right[0]) + abs(left[-1] - right[-1]) + abs(left[-1] - right[0])
    ans3 = abs(left[0] - right[0]) + abs(left[-1] - right[-1]) + abs(right[-1] - left[0])
    ans4 = abs(left[0] - right[0]) + abs(left[-1] - right[-1]) + abs(left[-1] - right[0])
    
    ans1 = abs(left[0]-right[0])
    for i in range(len(left)-1):
        ans1 += abs(left[i+1]-left[i])
    for i in range(len(right)-1):
        ans1 += abs(right[i+1]-right[i])
    
    ans2 = abs(left[0]-right[0])
    for i in range(len(left)-1):
        ans2 += abs(left[i+1]-left[i])
    for i in range(len(right)-1):
        ans2 += abs(right[i+1]-right[i])
    ans2 += abs(right[-1]-left[-1])
    
    ans3 = abs(left[0]-right[0])
    for i in range(len(left)-1):
        ans3 += abs(left[i+1]-left[i])
    for i in range(len(right)-1):
        ans3 += abs(right[i+1]-right[i])
    ans3 += abs(left[-1]-right[-1])
    
    
    ans = min(ans1,ans2,ans3)
    
    
    print(ans)