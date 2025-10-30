def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input())
    strings = [input().strip() for _ in range(N)]
    
    # Sort strings by their length (ascending)
    strings.sort(key=len)
    
    # Concatenate and print
    print("".join(strings))

# Call main to execute
main()