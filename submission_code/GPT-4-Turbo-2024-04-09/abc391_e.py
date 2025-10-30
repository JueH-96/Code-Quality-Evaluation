def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = data[1]
    
    # Calculate the length of the string
    length = 3 ** N
    
    # Function to compute the majority of a group of three
    def majority(x, y, z):
        if x == y or x == z:
            return x
        else:
            return y
    
    # Function to recursively calculate the majority string and count minimum changes
    def recursive_majority_and_changes(string, target):
        if len(string) == 1:
            if string == target:
                return 0
            else:
                return 1
        
        # Split the string into groups of three
        groups = [string[i:i+3] for i in range(0, len(string), 3)]
        new_string = []
        change_counts = []
        
        for group in groups:
            maj = majority(group[0], group[1], group[2])
            new_string.append(maj)
            
            # Calculate the minimum changes needed to make this group have a different majority
            if maj == '0':
                # Current majority is 0, count how many 1s needed to make majority 1
                count_1s = sum(1 for x in group if x == '1')
                changes_to_1 = 3 - count_1s if count_1s < 2 else 0
            else:
                # Current majority is 1, count how many 0s needed to make majority 0
                count_0s = sum(1 for x in group if x == '0')
                changes_to_0 = 3 - count_0s if count_0s < 2 else 0
            
            change_counts.append(changes_to_0 if maj == '1' else changes_to_1)
        
        # Recursively determine the minimum changes for the new string
        new_target = '0' if target == '1' else '1'
        min_changes = recursive_majority_and_changes(''.join(new_string), new_target)
        
        # Find the minimum change count to flip the final result
        min_change_to_flip = min(change_counts)
        
        # If we need to flip the final result, add the minimum change to flip one group
        if new_string[0] == target:
            return min_changes + min_change_to_flip
        else:
            return min_changes
    
    # Initial target is to flip the final result
    initial_target = '1' if A[0] == '0' else '0'
    result = recursive_majority_and_changes(A, initial_target)
    print(result)