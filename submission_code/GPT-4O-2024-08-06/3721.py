class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        online_status = [True] * numberOfUsers
        offline_until = [0] * numberOfUsers

        for event in events:
            event_type, timestamp_str, data = event
            timestamp = int(timestamp_str)

            # Update online status based on current timestamp
            for user_id in range(numberOfUsers):
                if not online_status[user_id] and timestamp >= offline_until[user_id]:
                    online_status[user_id] = True

            if event_type == "OFFLINE":
                user_id = int(data)
                online_status[user_id] = False
                offline_until[user_id] = timestamp + 60

            elif event_type == "MESSAGE":
                if data == "ALL":
                    for user_id in range(numberOfUsers):
                        mentions[user_id] += 1
                elif data == "HERE":
                    for user_id in range(numberOfUsers):
                        if online_status[user_id]:
                            mentions[user_id] += 1
                else:
                    ids = data.split()
                    for id_str in ids:
                        if id_str.startswith("id"):
                            user_id = int(id_str[2:])
                            mentions[user_id] += 1

        return mentions