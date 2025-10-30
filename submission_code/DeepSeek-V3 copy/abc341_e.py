import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    S = data[idx]
    idx += 1
    # Initialize the flip array
    flip = [0] * (N + 2)
    for _ in range(Q):
        type_q = int(data[idx])
        L = int(data[idx+1])
        R = int(data[idx+2])
        idx += 3
        if type_q == 1:
            flip[L] ^= 1
            flip[R+1] ^= 1
        else:
            # Calculate the number of flips for each character in the range
            # We need to find the parity of the number of flips for each character
            # Since flip is a difference array, the parity of flips up to i is the XOR of all flip[j] where j <= i
            # To find the parity for a range, we can precompute the prefix XOR
            # Initialize prefix_xor
            prefix_xor = [0] * (N + 2)
            for i in range(1, N+1):
                prefix_xor[i] = prefix_xor[i-1] ^ flip[i]
            # Now, for each character in the range [L, R], the parity is prefix_xor[i]
            # We need to check if all consecutive characters in the range have different parity
            # So, for each i from L to R-1, (S[i-1] ^ prefix_xor[i]) != (S[i] ^ prefix_xor[i+1])
            # Wait, no. The character at position i is S[i-1] ^ prefix_xor[i]
            # So, for i from L to R-1, (S[i-1] ^ prefix_xor[i]) != (S[i] ^ prefix_xor[i+1])
            # But since prefix_xor[i+1] = prefix_xor[i] ^ flip[i+1], and flip[i+1] is 0 or 1
            # So, S[i-1] ^ prefix_xor[i] != S[i] ^ (prefix_xor[i] ^ flip[i+1])
            # Simplify: S[i-1] ^ prefix_xor[i] != S[i] ^ prefix_xor[i] ^ flip[i+1]
            # Which is equivalent to S[i-1] != S[i] ^ flip[i+1]
            # So, for each i from L to R-1, S[i-1] != S[i] ^ flip[i+1]
            # But since flip[i+1] is 0 or 1, S[i] ^ flip[i+1] is S[i] flipped if flip[i+1] is 1
            # So, for each i from L to R-1, S[i-1] != (S[i] if flip[i+1] == 0 else (1 - S[i]))
            # So, we need to check for each i from L to R-1, S[i-1] != (S[i] if flip[i+1] == 0 else (1 - S[i]))
            # To implement this, we can precompute the effective character at each position
            # effective_char[i] = S[i-1] ^ prefix_xor[i]
            # So, for i from L to R-1, effective_char[i] != effective_char[i+1]
            # So, we can precompute effective_char for the entire string
            effective_char = [0] * (N + 1)
            for i in range(1, N+1):
                effective_char[i] = int(S[i-1]) ^ prefix_xor[i]
            # Now, check for i from L to R-1, effective_char[i] != effective_char[i+1]
            is_good = True
            for i in range(L, R):
                if effective_char[i] == effective_char[i+1]:
                    is_good = False
                    break
            if is_good:
                print("Yes")
            else:
                print("No")

if __name__ == "__main__":
    main()