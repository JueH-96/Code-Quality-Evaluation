def main():
    import sys
    input_data = sys.stdin.read().strip()
    # Read the integer N
    if input_data:
        try:
            n = int(input_data)
            # Concatenate the digit N n times
            result = str(n) * n
            print(result)
        except ValueError:
            print("Invalid input. Please provide an integer.")
            
if __name__ == "__main__":
    main()