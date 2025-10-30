N, T, A = map(int, input().split())

remaining = N - (T + A)

# If Takahashi gets all remaining votes
max_takahashi = T + remaining

# If Aoki gets all remaining votes  
max_aoki = A + remaining

# If Takahashi already has more votes than Aoki can possibly get
if T > max_aoki:
    print("Yes")
# If Aoki already has more votes than Takahashi can possibly get
elif A > max_takahashi:
    print("Yes") 
else:
    print("No")