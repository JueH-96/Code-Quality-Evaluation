def main():
    S = input().strip()
    # Extract the numeric part
    num_str = S[3:]
    num = int(num_str)
    
    # Check if num is in [1..315] or [317..349]
    if (1 <= num <= 315) or (317 <= num <= 349):
        print("Yes")
    else:
        print("No")

# Don't forget to call main()
main()