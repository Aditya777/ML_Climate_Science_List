
params="pumat21_30_0_100_1
pumat21_100_0_100_1
pumat42_30_0_100_1
pumat42_100_0_100_1
plasimt21_30_0_100_1
plasimt21_100_0_100_1
plasimt42_30_0_100_1
plasimt42_100_0_100_1"
for param in ${params}; do
    ffmpeg -i climnet${param}z_plus_u_%03d.png -b 1000000 video_${param}.mp4
done

