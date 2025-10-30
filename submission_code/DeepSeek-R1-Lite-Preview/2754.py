class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        positives = []
        negatives = []
        zero_count = 0
        
        for num in nums:
            if num > 0:
                positives.append(num)
            elif num < 0:
                negatives.append(num)
            else:
                zero_count += 1
        
        if positives:
            # Take all positives
            product = 1
            for num in positives:
                product *= num
            # Sort negatives in ascending order (most negative to least negative)
            negatives_sorted = sorted(negatives)
            if len(negatives_sorted) % 2 != 0:
                # Exclude the negative with the smallest absolute value
                negatives_sorted = negatives_sorted[:-1]
            for num in negatives_sorted:
                product *= num
            return product
        else:
            if negatives:
                negatives_sorted = sorted(negatives)
                if len(negatives_sorted) % 2 == 0:
                    # Take all negatives
                    product = 1
                    for num in negatives_sorted:
                        product *= num
                    return product
                else:
                    # Exclude the negative with the smallest absolute value
                    if len(negatives_sorted) > 1:
                        negatives_sorted = negatives_sorted[:-1]
                        product = 1
                        for num in negatives_sorted:
                            product *= num
                        return product
                    else:
                        # Only one negative, and no positives
                        return negatives_sorted[0]
            else:
                if zero_count >= 1:
                    return 0
        # In case of a single negative number
        return nums[0]