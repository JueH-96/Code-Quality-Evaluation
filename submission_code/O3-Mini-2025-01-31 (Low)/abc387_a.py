def main():
    import sys
    input_data = sys.stdin.read().strip()
    if input_data:
        parts = input_data.split()
        A = int(parts[0])
        B = int(parts[1])
        result = (A + B) ** 2
        print(result)
        
if __name__ == "__main__":
    main()