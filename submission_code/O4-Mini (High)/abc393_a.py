def main():
    s1, s2 = input().split()
    
    # Determine which oyster type causes trouble based on the reports:
    # Takahashi ate oysters {1,2}, Aoki ate oysters {1,3}.
    # If s1 == "sick", the bad type ∈ {1,2}, else ∉ {1,2}.
    # If s2 == "sick", the bad type ∈ {1,3}, else ∉ {1,3}.
    
    if s1 == "sick" and s2 == "sick":
        print(1)
    elif s1 == "sick" and s2 == "fine":
        print(2)
    elif s1 == "fine" and s2 == "sick":
        print(3)
    else:  # s1 == "fine" and s2 == "fine"
        print(4)

if __name__ == "__main__":
    main()