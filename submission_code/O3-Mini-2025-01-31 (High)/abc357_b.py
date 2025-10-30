def main():
    s = input().strip()
    
    # Count uppercase and lowercase letters
    count_upper = sum(1 for char in s if char.isupper())
    count_lower = sum(1 for char in s if char.islower())
    
    # Convert based on the condition given
    if count_upper > count_lower:
        print(s.upper())
    else:
        print(s.lower())

if __name__ == "__main__":
    main()