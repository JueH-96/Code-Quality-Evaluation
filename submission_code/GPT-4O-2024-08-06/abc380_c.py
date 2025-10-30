# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    
    # Step 1: Identify all 1-blocks
    blocks = []
    i = 0
    while i < N:
        if S[i] == '1':
            start = i
            while i < N and S[i] == '1':
                i += 1
            end = i - 1
            blocks.append((start, end))
        else:
            i += 1
    
    # Step 2: Reconstruct the string with the K-th block moved
    if K > 1:
        l_k_minus_1, r_k_minus_1 = blocks[K-2]
        l_k, r_k = blocks[K-1]
        
        # Create the new string
        result = []
        
        # Part 1: From start to r_{K-1}
        result.append(S[:r_k_minus_1 + 1])
        
        # Part 2: Add the K-th block after the (K-1)-th block
        result.append('1' * (r_k - l_k + 1))
        
        # Part 3: Fill the gap with zeros
        result.append('0' * (l_k - r_k_minus_1 - 1))
        
        # Part 4: From r_K + 1 to end
        result.append(S[r_k + 1:])
        
        # Join all parts and print the result
        print(''.join(result))
    else:
        # If K == 1, no need to move anything, just print S
        print(S)