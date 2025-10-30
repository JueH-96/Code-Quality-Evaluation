# Read input
X, Y = map(int, input().split())

# Calculate the difference in floors
diff = Y - X

# Takahashi uses stairs when:
# - Moving up two floors or less (0 < diff <= 2)
# - Moving down three floors or less (-3 <= diff < 0)
if (0 < diff <= 2) or (-3 <= diff < 0):
    print("Yes")
else:
    print("No")