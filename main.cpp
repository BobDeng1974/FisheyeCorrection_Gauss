#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat srcImg;
double DR=248;
double dr=20;

double thetaMax=89.0f/180.0f*CV_PI;

const int num_view=4;

double tmp_view[num_view+1];
double view_phi[num_view+1];

double f=DR/thetaMax;
double thetaMin=dr/f;

Point2f center;
int row;
int col;
cv::Mat mapx1;
cv::Mat mapy1;
float paras[] = { 0, 0 ,0};



Point2f find_center();
void sph2cart(const cv::Mat &phi, const cv::Mat &theta, cv::Mat &x, cv::Mat &y, cv::Mat &z);
void rotate3D(const float &alpha, const float &beta, const float &gamma, cv::Mat &Rx, cv::Mat &Ry, cv::Mat &Rz);
void cart2sph(const cv::Mat &x, const cv::Mat &y, const cv::Mat &z, cv::Mat &phi, cv::Mat &theta);
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);
void cal_subview(float(&pos1)[4], float(&pos2)[4], float(&paras)[3]);
void montage(Mat &mx1,Mat &my1,Mat &mx2,Mat &my2,int col,int row,float angle);
void cal_mapping(int &row, int &col,float angle,float paras1,float angle1 ,cv::Mat &mapx1, cv::Mat &mapy1,int number);


int main() {
    srcImg=imread("/Users/domino/img/fish_5.jpg",1);
    //初始化相关的视角范围，视线中心
    //确定四个视角区域边界
    for(int i=0;i<num_view;i++)
    {
        tmp_view[i]=i*2*CV_PI/num_view;
    }

    tmp_view[num_view]=view_phi[0]+2*CV_PI;
    //确定四个视角的中间线
    for(int i=0;i<num_view;i++)
    {
        view_phi[i]=((tmp_view[i+1]+tmp_view[i])/2);
    }

    //确定校正后图像的长宽，校正的角度大小
    float pos1[] = { float(thetaMax), float(thetaMax), float(thetaMin) ,0};
    float pos2[] = { CV_PI/4, CV_PI/2, CV_PI/4 ,CV_PI/2};
    cal_subview(pos1, pos2, paras);
    cout<<"校正四分图的长："<<paras[0]<<endl<<"校正四分图的宽："<<paras[1]<<endl<<"校正四分图的角度："<<-paras[2]*180/CV_PI<<endl;

    //计算映射
    row =srcImg.rows;
    col =srcImg.cols;
    center=find_center();

    mapx1.create(row, col, CV_32FC1);
    mapy1.create(row, col, CV_32FC1);

    //长,宽，变换过后的角度，变换过后的k宽度，共同区域角度，两个map，区域个数
    cal_mapping(row, col,paras[2],paras[1],paras[2]/2,mapx1, mapy1,2);

    //imshow("src",srcImg);
    Mat nimg1;

    cv::remap(srcImg, nimg1, mapx1, mapy1, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    imshow("gauss", nimg1);

    //imwrite("/Users/domino/Desktop/3.jpg",nimg1);
    cvWaitKey();
    return 0;
}

//找变化后图像的长宽
Point2f find_center()
{
    Point2f returnP;
    //2D到3D
    double pos1[]={CV_PI/4};
    double pos2[]={CV_PI/2};
    cv::Mat pos_theta(1, 1, CV_32FC1);
    cv::Mat pos_phi(1, 1, CV_32FC1);
    pos_theta.at<float>(0)=pos1[0];
    pos_phi.at<float>(0)=pos2[0];
    cv::Mat pos_x(1, 1, CV_32FC1, cv::Scalar(0));
    cv::Mat pos_y(1, 1, CV_32FC1, cv::Scalar(0));
    cv::Mat pos_r(1, 1, CV_32FC1);

    //成像模型（r=f*tan（theta））求R
    pos_r.at<float>(0, 0) = f * tan(pos_theta.at<float>(0, 0));

    polarToCart(pos_r, pos_phi, pos_x, pos_y);
    //横向为x轴，纵向为y轴
    returnP.x=pos_x.at<float>(0);
    returnP.y=pos_y.at<float>(0);
    cout<<"中点坐标（纵，横）："<<returnP<<endl;

    return returnP;
}
//长,宽，变换过后的角度，变换过后的宽度，共同区域角度，两个map，区域个数
//cal_mapping(row, col,paras[2],paras[1],paras[2]/2,mapx1, mapy1,2);
void cal_mapping(int &row, int &col,float angle,float paras1,float angle1 ,cv::Mat &mapx1, cv::Mat &mapy1,int number)
{
    cout<<"融合角度为："<<-angle1*180/CV_PI;

    Mat nX, nY;
    vector<float> p1,p2;
    for (int k = 0; k < number; k++)
    {
        //在三维旋转矩阵中 p1指绕x轴的旋转角度，p2指绕z轴旋转角度
        p2.push_back(view_phi[k]);
        p1.push_back(CV_PI/4);
        cv::Mat X, Y;
        meshgrid(cv::Range(1, col), cv::Range(1, row), X, Y);
        //直角坐标中心化
        cv::Mat c_X = X - col/2;
        cv::Mat c_Y = Y - row/2;
        c_X = c_X - center.x;
        c_Y = c_Y - center.y;
        //temp矩阵由0，1组成 确定显示位置
        cv::Mat temp(row, col, CV_32FC1, cv::Scalar(0));

        Mat tmprou,tmpphi;
        cartToPolar(c_X,c_Y,tmprou,tmpphi);
        //临时矩阵，存储有效位置
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
            {
                //根据两个分图来给temp赋值确定显示范围
                switch (k)
                {
                    case 1:if(tmpphi.at<float>(i, j)< 3*CV_PI/2 - angle && tmpphi.at<float>(i, j)> 3*CV_PI/2+angle1&& tmprou.at<float>(i,j)<paras1)
                        {
                            temp.at<float>(i, j) = 1;
                        }
                        break;
                    case 0:if(tmpphi.at<float>(i, j)< 3*CV_PI/2-angle1 && tmpphi.at<float>(i, j)> 3*CV_PI/2 + angle && tmprou.at<float>(i,j)<paras1)
                        {
                            temp.at<float>(i, j) = 1;
                        }

                        break;
                }
            }
        c_X=c_X+center.x;
        c_Y=c_Y+center.y;
        cv::Mat phi, rou;
        cv::cartToPolar(c_X, c_Y, rou, phi);
        //转化为二角球坐标（phi，theta）
        cv::Mat theta(row, col, CV_32FC1);
        for (int j = 0; j < col; j++)
            for (int i = 0; i < row; i++)
                theta.at<float>(i, j) = atan2(rou.at<float>(i, j), f);

        cv::Mat XX(row, col, CV_32FC1, cv::Scalar(0));
        cv::Mat YY(row, col, CV_32FC1, cv::Scalar(0));
        cv::Mat ZZ(row, col, CV_32FC1, cv::Scalar(0));
        //二角球坐标转化为标准球坐标（x，y，z）
        sph2cart(phi, theta, XX, YY, ZZ);
        //求旋转矩阵分别绕 x轴，y轴，z轴的旋转矩阵
        cv::Mat Rx(3, 3, CV_32FC1);
        cv::Mat Ry(3, 3, CV_32FC1);
        cv::Mat Rz(3, 3, CV_32FC1);

        rotate3D(p1.at(k), 0, p2.at(k), Rx, Ry, Rz);
        /*旋转*/
        cv::Mat XYZ(row*col, 3, XX.type());
        cv::Mat TXX = XX.reshape(0, row*col);
        cv::Mat TYY = YY.reshape(0, row*col);
        cv::Mat TZZ = ZZ.reshape(0, row*col);
        TXX.copyTo(XYZ.col(0));
        TYY.copyTo(XYZ.col(1));
        TZZ.copyTo(XYZ.col(2));
        //乘上旋转矩阵，进行三维旋转
        XYZ = XYZ *Rx* Rz;
        XYZ.col(0).copyTo(TXX);
        XYZ.col(1).copyTo(TYY);
        XYZ.col(2).copyTo(TZZ);

        cv::Mat nphi(row, col, CV_32FC1, cv::Scalar(0));
        cv::Mat ntheta(row, col, CV_32FC1, cv::Scalar(0));
        cart2sph(XX, YY, ZZ, nphi, ntheta);
        //运用鱼眼成像模型（r = f * theta）转换到二维极坐标
        cv::Mat nrou(row, col, CV_32FC1);
        for (int j = 0; j < col; j++)
        {
            for (int i = 0; i < row; i++)
            {
                nrou.at<float>(i, j) =	f *(ntheta.at<float>(i, j));
            }
        }

        polarToCart(nrou, nphi, nX, nY);
        //当为 两张图时候确定最后合成图第一张和第二张需要旋转的角度
        if(number==2)
        {
            CvPoint2D32f center1=CvPoint2D32f(nX.cols/2+0,nX.rows/2+center.y);
            Mat rotateMat;
            double scale=1;
            switch (k)
            {
                case 0:rotateMat=getRotationMatrix2D(center1,-angle*180/CV_PI/2,scale);break;
                case 1:rotateMat=getRotationMatrix2D(center1,angle*180/CV_PI/2,scale);break;
            }
            warpAffine(nX,nX,rotateMat,nX.size());
            warpAffine(nY,nY,rotateMat,nX.size());
        }

        nX=nX.mul(temp);nY=nY.mul(temp);

        Mat nimg1;
        remap(srcImg, nimg1, nX+col/2, nY+row/2, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        if(k==0)
        {
            imshow("111",nimg1);
            //imwrite("/Users/domino/Desktop/1.jpg",nimg1);
        }

        if(k==1)
        {
            imshow("222",nimg1);
            //imwrite("/Users/domino/Desktop/2.jpg",nimg1);
        }

        switch (k)
        {
            case 0:
                mapx1 += nX;
                mapy1 += nY;
                break;
            case 1:
                mapx1=mapx1+col/2;
                mapy1=mapy1+row/2;
                nX=nX+col/2;
                nY=nY+row/2;
                montage(mapx1,mapy1,nX,nY,col,row,angle1);
                break;
        }
    }
}

// montage(mapx1,mapy1,nX,nY,col,row,angle1);
void montage(Mat &mx1,Mat &my1,Mat &mx2,Mat &my2,int col,int row,float angle)
{
    Mat X, Y;
    meshgrid(cv::Range(1, col), cv::Range(1, row), X, Y);
    Mat c_X = X - col/2;
    Mat c_Y = Y - row/2 - center.y;
    Mat phi, rou;
    cartToPolar(c_X, c_Y, rou, phi);

    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
        {
            //double theta = atan2(i, j);//极坐标转换得到角度theta
            //cout<<phi.at<float>(i, j)<<endl;
            //cout<<3*CV_PI/2-(angle/2)<<"-"<<3*CV_PI/2 + (angle/2)<<endl;
            //cout<<angle;
            if(phi.at<float>(i, j)<= 3*CV_PI/2-(angle/2) && phi.at<float>(i, j)>= 3*CV_PI/2 + (angle/2))
            {
                double b = 3*CV_PI/2 + (-angle/2);
                double c = 3*CV_PI/2-(-angle/2);
                double a =b-c;

                double e= 2.718281828;
                double mask=pow(e,-( (phi.at<float>(i, j)-(a+c))*(phi.at<float>(i, j)-(a+c)))/2*30);
                //cout << phi.at<float>(i, j) << "--" << y << endl;

                mx1.at<float>(i,j)=(1-mask)*mx1.at<float>(i,j)+(mask)*mx2.at<float>(i,j);
                my1.at<float>(i,j)=(1-mask)*my1.at<float>(i,j)+(mask)*my2.at<float>(i,j);
            }
            else if(phi.at<float>(i, j)>= 3*CV_PI/2+(angle/2) )
            {
                mx1.at<float>(i,j)=mx2.at<float>(i,j);
                my1.at<float>(i,j)=my2.at<float>(i,j);
            }
        }

}
void cal_subview(float(&pos1)[4], float(&pos2)[4], float(&paras)[3])
{
    //2D到3D
    cv::Mat pos_theta(4, 1, CV_32FC1, pos1);
    cv::Mat pos_phi(4, 1, CV_32FC1, pos2);
//  cout<<pos_theta*180/CV_PI<<endl;
    cv::Mat pos_x(4, 1, CV_32FC1, cv::Scalar(0));
    cv::Mat pos_y(4, 1, CV_32FC1, cv::Scalar(0));
    cv::Mat pos_z(4, 1, CV_32FC1, cv::Scalar(0));
    sph2cart(pos_phi, pos_theta, pos_x, pos_y, pos_z);

    //获取旋转矩阵
    cv::Mat Rx(3, 3, CV_32FC1);
    cv::Mat Ry(3, 3, CV_32FC1);
    cv::Mat Rz(3, 3, CV_32FC1);
    rotate3D(CV_PI / 4,0, CV_PI / 4, Rx, Ry, Rz);

    //获取球面上点的两个角度
    cv::Mat stdxyz(pos_x.rows, 3, pos_x.type());
    pos_x.copyTo(stdxyz.col(0));
    pos_y.copyTo(stdxyz.col(1));
    pos_z.copyTo(stdxyz.col(2));
    stdxyz = stdxyz * Rz* Rx;
    stdxyz.col(0).copyTo(pos_x);
    stdxyz.col(1).copyTo(pos_y);
    stdxyz.col(2).copyTo(pos_z);
    cart2sph(pos_x, pos_y, pos_z, pos_phi, pos_theta);

    cv::Mat pos_r(pos_x.rows, pos_x.cols, CV_32FC1);

    //成像模型（r=f*tan（theta））求R
    for (int i = 0; i < pos_r.rows; i++)
    {
        pos_r.at<float>(i, 0) = f * tan(pos_theta.at<float>(i, 0));
    }


    polarToCart(pos_r, pos_phi, pos_x, pos_y);
    //paras[0] paras[1]为纵向y坐标的长度,paras[2]为校正后二分之一部分的角度大小
    paras[0] = 2.0f * abs(pos_x.at<float>(1, 0));
    paras[1] = abs(pos_y.at<float>(0, 0) - pos_y.at<float>(3, 0));
    paras[2] = 2*atan((pos_x.at<float>(1)-pos_x.at<float>(0))/(pos_y.at<float>(1)-pos_y.at<float>(3)));
    //cout<<paras[0]<<"--"<<paras[1]<<"--"<<paras[2]<<endl;
}


void sph2cart(const cv::Mat &phi, const cv::Mat &theta, cv::Mat &x, cv::Mat &y, cv::Mat &z)
{
    for (int j = 0; j < phi.cols; j++)
        for (int i = 0; i < phi.rows; i++)
        {
            if (!(phi.at<float>(i, j) == 0 && theta.at<float>(i, j) == 0))
            {
                x.at<float>(i, j) = std::sin(theta.at<float>(i, j)) * std::cos(phi.at<float>(i, j));
                y.at<float>(i, j) = std::sin(theta.at<float>(i, j)) * std::sin(phi.at<float>(i, j));
                z.at<float>(i, j) = std::cos(theta.at<float>(i, j));
            }
        }
}

void rotate3D(const float &alpha, const float &beta, const float &gamma, cv::Mat &Rx, cv::Mat &Ry, cv::Mat &Rz)
{
    Rx.at<float>(0, 0) = 1; Rx.at<float>(0, 1) = 0;			  Rx.at<float>(0, 2) = 0;
    Rx.at<float>(1, 0) = 0; Rx.at<float>(1, 1) = cos(alpha);  Rx.at<float>(1, 2) = sin(alpha);
    Rx.at<float>(2, 0) = 0; Rx.at<float>(2, 1) = -sin(alpha); Rx.at<float>(2, 2) = cos(alpha);

    Ry.at<float>(0, 0) = cos(beta); Ry.at<float>(0, 1) = 0;	Ry.at<float>(0, 2) = -sin(beta);
    Ry.at<float>(1, 0) = 0;			Ry.at<float>(1, 1) = 1; Ry.at<float>(1, 2) = 0;
    Ry.at<float>(2, 0) = sin(beta); Ry.at<float>(2, 1) = 0; Ry.at<float>(2, 2) = cos(beta);

    Rz.at<float>(0, 0) = cos(gamma);  Rz.at<float>(0, 1) = sin(gamma); Rz.at<float>(0, 2) = 0;
    Rz.at<float>(1, 0) = -sin(gamma); Rz.at<float>(1, 1) = cos(gamma); Rz.at<float>(1, 2) = 0;
    Rz.at<float>(2, 0) = 0;			  Rz.at<float>(2, 1) = 0;		   Rz.at<float>(2, 2) = 1;
}
void cart2sph(const cv::Mat &x, const cv::Mat &y, const cv::Mat &z, cv::Mat &phi, cv::Mat &theta)
{
    for (int j = 0; j < x.cols; j++)
        for (int i = 0; i < x.rows; i++)
        {
            float r =  sqrt(x.at<float>(i, j)*x.at<float>(i, j) + y.at<float>(i, j)*y.at<float>(i, j) + z.at<float>(i, j)*z.at<float>(i, j));
            if (r != 0)
            {
                phi.at<float>(i, j) = atan2(y.at<float>(i, j), x.at<float>(i, j));
                theta.at<float>(i, j) = acos(z.at<float>(i, j) / r);
            }
        }
}
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)
{
    std::vector<float> t_x, t_y;
    for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(float(i));
    for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(float(j));
    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}



