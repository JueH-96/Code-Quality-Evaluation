def main():
    import sys

    data = sys.stdin.read().strip().split()
    A, B = map(int, data)
    
    s = A + B
    for i in range(10):
        if i != s:
            print(i)
            return

# Call the main function
main()