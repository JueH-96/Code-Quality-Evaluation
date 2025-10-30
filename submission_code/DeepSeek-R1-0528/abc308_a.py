def main():
    data = input().split()
    nums = list(map(int, data))
    
    non_decreasing = True
    for i in range(1, len(nums)):
        if nums[i] < nums[i-1]:
            non_decreasing = False
            break
            
    valid_range = True
    for num in nums:
        if num < 100 or num > 675:
            valid_range = False
            break
            
    multiple_of_25 = True
    for num in nums:
        if num % 25 != 0:
            multiple_of_25 = False
            break
            
    if non_decreasing and valid_range and multiple_of_25:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()