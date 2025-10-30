def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)
    
    result = (A ** B) + (B ** A)
    print(result)

# Do not forget to call main() at the end
main()