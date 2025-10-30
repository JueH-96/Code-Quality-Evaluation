# Read input string
S = input().strip()
N = len(S)

# Initialize counter for valid sequences
count = 0

# Try all possible starting positions i
for i in range(N-2):
    # Try all possible intervals d
    for d in range(1, (N-i)//2 + 1):
        # Calculate j and k positions
        j = i + d
        k = j + d
        
        # Check if k is within string bounds
        if k >= N:
            continue
            
        # Check if characters at positions i,j,k form 'ABC'
        if S[i] == 'A' and S[j] == 'B' and S[k] == 'C':
            count += 1

# Print result
print(count)