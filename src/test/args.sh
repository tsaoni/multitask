BUCKET_NAME=testbucket
OBJECT_NAME=testworkflow-2.0.1.jar
TARGET_LOCATION=/opt/test/testworkflow-2.0.1.jar
HI=hi

JSON_STRING=$( jq -n \
                  --arg bn "$BUCKET_NAME" \
                  --arg on "$OBJECT_NAME" \
                  --arg tl "$TARGET_LOCATION" \
                  '{
                        xx: {
                            bucketname: $bn, 
                            objectname: $on, 
                            targetlocation: $tl
                        }
                    }' )
OUTER=$( jq -n \
            --arg xx "$HI" \
            --arg yy "$JSON_STRING" \
            '{
                xx: $yy
                }' )
echo "${JSON_STRING}"
echo "${OUTER}"