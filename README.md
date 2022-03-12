# Face-Swapping
In this Project I have tried to do face swapping based on face landmarks. 
## Source and Destination Images
![alt text](./images/Im387_resize_.png)
![alt text](./images/Im386_resize_.png)
## Result
![alt text](.result.jpg)
### Steps:
  - Take two iamges
  -  Find landmark points of two images
  -  Triangulation on source image
  -  Triangulation on destination image
  -  Extract and warp trangles
  -  Link the warped trangles together
  -  Replace the face on destination image
