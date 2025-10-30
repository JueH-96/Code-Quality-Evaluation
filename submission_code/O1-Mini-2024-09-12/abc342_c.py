# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr]); ptr += 1
    S = data[ptr]; ptr +=1
    Q = int(data[ptr]); ptr +=1
    operations = []
    for _ in range(Q):
        c = data[ptr]; d = data[ptr+1]; ptr +=2
        operations.append( (c,d) )
    
    mapping = [chr(ord('a') + i) for i in range(26)]
    rev = [ [chr(ord('a') + i)] for i in range(26)]
    
    for c, d in operations:
        ci = ord(c) - ord('a')
        di = ord(d) - ord('a')
        # Replace all characters in rev[ci] to d
        for x in rev[ci]:
            xi = ord(x) - ord('a')
            mapping[xi] = d
        # Add all these characters to rev[di]
        rev[di].extend(rev[ci])
        # Clear rev[ci]
        rev[ci] = []
    
    # Build translation table
    # Create a list for quick access
    # Convert S to list for speed
    S_list = list(S)
    for i in range(N):
        S_list[i] = mapping[ord(S_list[i]) - ord('a')]
    print(''.join(S_list))

if __name__ == "__main__":
    main()