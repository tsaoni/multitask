BUCKET_NUM=1
OBJECT_NAME=testworkflow-2.0.1.jar
TARGET_LOCATION=/opt/test/testworkflow-2.0.1.jar

JSON_STRING=$( jq -n \
                  --arg bn "$BUCKET_NUM" \
                  --arg on "$OBJECT_NAME" \
                  --arg tl "$TARGET_LOCATION" \
                  '{bucketname: $bn, objectname: $on, targetlocation: $tl}' )

python a.py \
 --try_dict "$JSON_STRING" \
 --other 1
