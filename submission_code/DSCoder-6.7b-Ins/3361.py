class Solution:
    def findLatestTime(self, s: str) -> str:
        # Replace "?", convert to int and sort the list
        times = sorted(int(i) if i != "?" else 99 for i in s.replace(":", ""))
        # Find the maximum time
        max_time = max(times)
        # If the maximum time is 99, replace it with 59
        if max_time == 99:
            times[times.index(max_time)] = 59
        # If the maximum time is 49, replace it with 19
        elif max_time == 49:
            times[times.index(max_time)] = 19
        # If the maximum time is 39, replace it with 29
        elif max_time == 39:
            times[times.index(max_time)] = 29
        # If the maximum time is 29, replace it with 39
        elif max_time == 29:
            times[times.index(max_time)] = 39
        # If the maximum time is 19, replace it with 49
        elif max_time == 19:
            times[times.index(max_time)] = 49
        # Convert the list back to string
        time_str = "".join(str(i).zfill(2) for i in times)
        # Return the time string in 12-hour format
        return f"{time_str[:2]}:{time_str[2:]}"