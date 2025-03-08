DetectionResult(
    detections=[
        Detection(
            bounding_box=BoundingBox(origin_x=72, origin_y=162, width=252, height=191), 
            categories=[
                Category(index=None, score=0.7797361016273499, display_name=None, category_name='cat')
            ], keypoints=[]), 
        Detection(
            bounding_box=BoundingBox(origin_x=303, origin_y=27, width=249, height=345), 
            categories=[
                Category(index=None, score=0.7622121572494507, display_name=None, category_name='dog')
            ], keypoints=[])])


print(detection_result.detections[0].categories[0].category_name) #cat
print(detection_result.detections[1].categories[0].category_name) #dog

DetectionResult(
    detections=[
        Detection(
            bounding_box=BoundingBox(origin_x=1494, origin_y=108, width=1005, height=2445), 
            categories=[Category(index=None, score=0.6220653057098389, display_name=None, category_name='person')
                                                                                                        ], keypoints=[]), 
        Detection(bounding_box=BoundingBox(origin_x=323, origin_y=137, width=918, height=2405), categories=[
            Category(index=None, score=0.5352797508239746, display_name=None, category_name='person')
                                                                                                            ], keypoints=[])])