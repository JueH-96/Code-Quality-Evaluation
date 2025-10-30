# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    first_pipe = S.index('|')
    second_pipe = S.rindex('|')
    
    result = S[:first_pipe] + S[second_pipe + 1:]
    print(result)

if __name__ == "__main__":
    main()