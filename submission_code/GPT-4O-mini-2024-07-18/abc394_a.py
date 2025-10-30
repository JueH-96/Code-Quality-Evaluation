def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Filter the string to keep only '2'
    result = ''.join(char for char in S if char == '2')
    
    print(result)

if __name__ == "__main__":
    main()