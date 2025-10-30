def main():
    import sys
    
    R, G, B = map(int, sys.stdin.readline().split())
    C = sys.stdin.readline().strip()

    if C == "Red":
        answer = min(G, B)
    elif C == "Green":
        answer = min(R, B)
    else:  # C == "Blue"
        answer = min(R, G)
    
    print(answer)

# Do not remove the call to main()
main()