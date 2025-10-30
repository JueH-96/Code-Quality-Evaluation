import sys

def main():
    S = sys.stdin.readline().strip()
    
    # If the first character is the unique one, it appears only once
    if S.count(S[0]) == 1:
        print(1)
        return
    
    # Otherwise, the unique character is the first one that differs from S[0]
    for i, ch in enumerate(S):
        if ch != S[0]:
            print(i + 1)           # 1-based index
            return

if __name__ == "__main__":
    main()