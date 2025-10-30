# Read the input integer N
N = int(input())

# Convert N to string to easily count digits
s = str(N)

# Check the conditions for digits 1, 2, and 3
if s.count('1') == 1 and s.count('2') == 2 and s.count('3') == 3:
    print("Yes")
else:
    print("No")