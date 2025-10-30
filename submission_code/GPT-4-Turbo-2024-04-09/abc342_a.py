def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Find the character that appears only once
    for i in range(len(S)):
        if S[i] != S[0]:
            print(i)
            break
        elif S[i] != S[-1]:
            print(i)
            break

if __name__ == "__main__":
    main()