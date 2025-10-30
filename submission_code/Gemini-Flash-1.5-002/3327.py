class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones = sum(nums)
        if ones >= k:
            return 0

        ans = float('inf')
        for start_index in range(n):
            moves = 0
            temp_nums = nums[:]
            
            if temp_nums[start_index] == 1:
                temp_nums[start_index] = 0
                ones -=1
            else:
                moves +=1
                temp_nums[start_index] = 1
                ones +=1


            changes_made = 0
            
            
            
            while ones < k:
                found = False
                for i in range(n):
                    if temp_nums[i] == 0:
                        if changes_made < maxChanges:
                            temp_nums[i] = 1
                            ones += 1
                            changes_made += 1
                            moves += 1
                            found = True
                            break
                        else:
                            
                            break
                if not found:
                    
                    found_swap = False
                    for i in range(n):
                        if temp_nums[i] == 1:
                            if i > 0 and temp_nums[i-1] == 0:
                                temp_nums[i], temp_nums[i-1] = temp_nums[i-1], temp_nums[i]
                                if i-1 == start_index:
                                    temp_nums[i-1] = 0
                                    ones -=1
                                moves += 1
                                found_swap = True
                                break
                            elif i < n -1 and temp_nums[i+1] == 0:
                                temp_nums[i], temp_nums[i+1] = temp_nums[i+1], temp_nums[i]
                                if i+1 == start_index:
                                    temp_nums[i+1] = 0
                                    ones -=1
                                moves += 1
                                found_swap = True
                                break
                    if not found_swap:
                        break

            ans = min(ans, moves)

        return ans