n, t, a = map(int, input().split())

remaining_votes = n - t - a

if t > (n // 2) :
    print("Yes")
elif (t + remaining_votes) <= (n // 2):
    print("No")
elif (a > (n // 2)):
    print("Yes")
elif (a + remaining_votes) <= (n // 2):
    print("No")
elif t > (n - (n//2)):
    print("Yes")
elif a > (n - (n//2)):
    print("Yes")
else:
    print("No")