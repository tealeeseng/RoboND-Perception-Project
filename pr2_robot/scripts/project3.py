#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
# from builtins import object
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    pcl.save(cloud, 'init_cloud.pcd')

    
    # Statistical Outlier Filtering
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(20)
    x=0.1
    outlier_filter.set_std_dev_mul_thresh(x)
    stat_cloud_filter = outlier_filter.filter()
    pcl.save(stat_cloud_filter, 'stat.pcd')



    # TODO: Voxel Grid Downsampling
    vox = stat_cloud_filter.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    pcl.save(cloud_filtered, 'vox.pcd')


    #PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.63
    axis_max = 1
    passthrough.set_filter_limits(axis_min, axis_max)
    passthrough = passthrough.filter()

    passthrough = passthrough.make_passthrough_filter()
    filter_axis = 'x'
    passthrough.set_filter_field_name(filter_axis)
    x_axis_min = 0.35
    x_axis_max = 1
    passthrough.set_filter_limits(x_axis_min, x_axis_max)
    cloud_filtered = passthrough.filter()

    pcl.save(cloud_filtered, 'passthrough.pcd')
    #
    # rospy.loginfo('saved.');
    # exit();

    #RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coeffcients = seg.segment()

    # TODO: Extract inliers and outliers
    cloud_objects = cloud_filtered.extract(inliers, negative = True)
    cloud_background = cloud_filtered.extract(inliers, negative = False)

    pcl.save(cloud_objects, 'segmentation.pcd')


    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.03) # 0..03
    ec.set_MinClusterSize(30)   # 30
    ec.set_MaxClusterSize(800) #800
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    pcl.save(cluster_cloud, 'Euclidean.pcd')

    rospy.loginfo('saved.');
    # exit();


    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_backgroud = pcl_to_ros(cloud_background)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_background_pub.publish(ros_cloud_backgroud)
    pcl_cluster_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects_list = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        christ = compute_color_histograms(ros_cluster, using_hsv=True) #bins of 128
        normals = get_normals(ros_cluster)
        nhists=compute_normal_histograms(normals) # bins of 128
        feature = np.concatenate((christ,nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2]+=.4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects_list.append(do)

    # Publish the list of detected objects

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects_list)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    test_num = 2
    output = []

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')



    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for param in object_list_param:
        object_name = param['name']
        object_group = param['group']


        for object in object_list:
            if object_name != object.label:
                continue

            # labels.append(object.label)
            ros_scene_num = Int32()
            ros_scene_num.data = test_num

            ros_object_name = String()
            ros_object_name.data = object_name

            # TODO: Assign the arm to be used for pick_place
            ros_arm = String()
            if object_group == 'red':
                ros_arm.data='left'
            else:
                ros_arm.data='right'


            # TODO: Get the PointCloud for a given object and obtain it's centroid
            points_arr = ros_to_pcl(object.cloud).to_array()
            centroids = np.mean(points_arr, axis=0)[:3] #np.asscalar
            centroid = [np.asscalar(x) for x in centroids]

            pick_pose = Pose()
            pick_pose.position.x = centroid[0]
            pick_pose.position.y = centroid[1]
            pick_pose.position.z = centroid[2]

        # TODO: Create 'place_pose' for the object
        ros_place_pose = Pose()
        for box in dropbox_param:
            if box['group'] == object_group:
                ros_place_pose.position.x = box['position'][0]
                ros_place_pose.position.y = box['position'][1]
                ros_place_pose.position.z = box['position'][2]

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        yaml_dict = make_yaml_dict(ros_scene_num, ros_arm, ros_object_name, pick_pose, ros_place_pose)
        output.append(yaml_dict)






        # Wait for 'pick_place_routine' service to come up
        # rospy.wait_for_service('pick_place_routine')
        #
        # try:
        #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
        #
        #     # TODO: Insert your message variables to be sent as a service request
        #     resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)
        #
        #     print ("Response: ",resp.success)
        #
        # except rospy.ServiceException, e:
        #     print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml('test-'+str(test_num)+'.yaml', output)
    rospy.loginfo("yaml ok.")



if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('project3')

    # models = ['biscuits', 'soap', 'soap2', 'book', 'glue', 'sticky_notes', 'snacks', 'eraser']


    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points",
                               pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_background_pub = rospy.Publisher("/pcl_background", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)



    # TODO: Load Model From disk
    model = pickle.load(open('model-200.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
