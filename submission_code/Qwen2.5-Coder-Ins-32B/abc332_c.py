# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    S = input[2]
    
    logo_needed = 0
    plain_used = 0
    logo_used = 0
    
    for day in S:
        if day == '1':
            if plain_used < M:
                plain_used += 1
            else:
                logo_used += 1
        elif day == '2':
            logo_used += 1
        
        if day == '0':
            plain_used = 0
            logo_used = 0
    
    print(max(0, logo_used - (M - plain_used)))

if __name__ == "__main__":
    main()