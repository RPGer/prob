#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>

// #include <dirent.h>

#include "rand_expr.h"
#include "eiterator.h"

using namespace std;
using namespace cv;

#define HIST_SIZE (18 + 26)
void calcHSHist(Mat im, Mat &hist)
{
    double *dhist = new double[HIST_SIZE];
    Mat hsv;
    cvtColor(im, hsv, COLOR_BGR2HSV);
    for(int y = 0; y < im.rows; y++) {
        unsigned char *ptr = im.ptr<uchar>(y);
        for(int x = 0; x < im.cols; x++) {
            dhist[ptr[x * 3] / 10]++;
            dhist[18 + ptr[x * 3 + 1] / 10]++;
        }
    }
    for(int i = 0; i < HIST_SIZE; i++) {
        dhist[i] /= im.rows * im.cols;
    }
    hist = Mat(Size(HIST_SIZE, 1), CV_64FC1, dhist);
}

int main(int argc, const char * argv[])
{
    // string s = typeid(YourClass).name()
//    srand((unsigned int)1232453); rand();
    srand(time(NULL)); rand();
    Environment env;

    //Mat img = imread("fruits.jpg"/*, IMREAD_GRAYSCALE*/);
    //resize(img, img, Size(80, 80));

    /*List *imgs = new List();

    DIR *dir = opendir("pics");
    struct dirent *result;
    while((result = readdir(dir))) {
        if(!strstr(result->d_name, ".jpg") && !strstr(result->d_name, ".JPEG") ) continue;
        cout << result->d_name << endl;
        Mat im = imread(string("pics/") + string(result->d_name));
        Mat hist;
        calcHSHist(im, hist);
        *imgs << V(hist);
    }*/


    ExpressionP p = make_shared<Block>();
    //Symbol f("f"), x("x"), y("y"), n("n"), im("im"), imx("imx"), imy("imy");
    //Symbol nClust("nClust"), centers("centers"), center("center"), im("im");
    //cv::Mat img = Mat::zeros(127, 128, CV_8UC1);
    //circle(img, Point(30, 30), 20, Scalar(255, 255, 255));

    cv::Mat img = imread("pic31.png", IMREAD_GRAYSCALE);

	//cv::Mat img = imread("pic2_all.png", IMREAD_GRAYSCALE);
    cv::resize(img, img, Size(img.cols / 5, img.rows / 5));
    img = 255 - img;
    /*cv::Mat imgx, imgy;
    cv::GaussianBlur(img, img, Size(17, 17), 0);
    cv::Sobel(img, imgx, CV_32FC1, 1, 0);
    cv::Sobel(img, imgy, CV_32FC1, 0, 1);
    imgx = abs(imgx) + abs(imgy);
    imgx.convertTo(img, CV_8UC1);*/

#if 0
    namedWindow("original");
    imshow("original", img);

	//поиск эритроцитов
    Symbol im, imc, im0, n, circs, x, y, circ, r0;
    p = sGPQuery(List() << include_libdefs()
                       //<< Define(n, V(27))
                       << Define(n, V(6))
                       << Define(circs,
                                 ::repeat(n, Lambda0(List() << V(Drawer::shape_circle)
                                                          << RndInt(img.cols + 10)-5
                                                          << RndInt(img.rows + 10)-5
                                // doesn't work correctly: r0 is substituted many times
                                // and then is changed independently
                                                     << /*RndInt(15) + 5*/ V(6)+RndInt(12)
                                                          << V(168) << V(-1))))
                       /*<< Define(im0, foldr(Lambda(circ, im,
                                       DrawCircle(im, circ[0], circ[1], circ[2], V(255), V(-1))),
                                circs, V(Mat::zeros(img.rows, img.cols, CV_8UC1))))*/
                       << Define(im0, Drawer(V(Mat::zeros(img.rows, img.cols, CV_8UC1)), circs))
                       //<< Define(imc, ::GaussianBlur(im0, V(13), V(0)) * 5)
                       << im0 //List(im0, circs)
                       << Log(MatDiff2(im0/*c*/, V(img))) * V(img.cols * img.rows)
                       )->reeval(&env, NULL, 0.005);
#endif

#if 1
	//задача о подмножествах
	Symbol n, xs, ws, makews, sum, summ, xxs, wws;
	p = sGPQuery(List() << include_libdefs()
				<< Define(xs, List() << V(9568) <<  V(5716) <<  V(8382) << V(7900) << V(-5461) << V(5087) << V(1138) << V(-1111) << V(-9695) << V(-5468) << V(6345) << V(-1473) << V(-7521) << V(-4323) << V(9893) << V(-9032) << V(-4715) << V(3699) << V(5104) << V(1551))
				<< Define(n, Length(xs))
				<< Define(ws, ::repeat(n, Lambda0(Flip())))
				<< Define(summ, Lambda(xxs, wws, 
									If(Nullp(xxs), V(0.), 
										If(Car(wws), Car(xxs), V(0)) + summ(Cdr(xxs), Cdr(wws)) )
									))
				<< Define(sum, summ(xs, ws))
				<< ws
                << sum*sum
                )->reeval(&env, NULL, 1);
#endif

#if 0
	//задача интерпол€ции кривой
	Symbol xs, ys, x, ws, calcpoly, generate, n, sigma, makews, wn, ysgen, ysg, yso, sum;
	p = sGPQuery(List() << include_libdefs()
				<< Define(xs, List() << V(0) << V(1) << V(2) << V(3) << V(4))
				<< Define(ys, List() << V(0.05) << V(4.9) << V(14.06) << V(29.99) << V(43.97))
				<< Define(calcpoly, Lambda(x, ws,
										If(Nullp(ws), V(0.),
											Car(ws)+x*calcpoly(x, Cdr(ws))
											)))
				<< Define(generate, Lambda(xs, ws,
										If(Nullp(xs), List(),
											Cons(generate(Cdr(xs), ws), calcpoly(Car(xs), ws))
											)))
				<< Define(makews, Lambda(n, sigma, 
										If(n<=0, List(), 
											Cons(makews(n-1, sigma), Gaussian(V(0.), sigma))
											)))
				<< Define(wn, RndInt(Length(xs))+1)
				<< Define(ws, makews(wn, V(10.)))
				<< Define(ysgen, generate(xs, ws))
				<< Define(sum, Lambda(ysg, yso,
									If(Nullp(ysg), V(0.),
										(Car(ysg)-Car(yso))*(Car(ysg)-Car(yso))+sum(Cdr(ysg),Cdr(yso)))))
				<< wn
				<< sum(ysgen, ys)
				)->reeval(&env, NULL, 0.005);

#endif

#if 0
	//задача коммиво€жЄра
	Symbol pts, plength, fst, snd, x, msqr, n, genn, gennn, nn, baseway, goalway, pp, indOf, bs, gint,
		   llst, ffst, ssnd, delinlist, ind, lst, i, head, merge, xs, ys, por, porr, porrr, gg, fulllen,
		   zx, cx, lCar, lCdr, ccopy, tmp, j, ans, delRef, vs, k;
	p = sGPQuery(List() << include_libdefs()
				<< Define(pts, List() << (List() << V(1.) << V(1.)) << (List() << V(6.) << V(3.)) << (List() << V(4.) << V(4.))
									  << (List() << V(0.) << V(9.)) << (List() << V(6.) << V(8.)) << (List() << V(0.) << V(0.)))
				<< Define(n, Length(pts))
				//генераци€ чисел от 1 до n
				<< Define(genn, Lambda(nn, If(nn<=1, List() << V(1), Cons(genn(nn-1), nn) )))
				<< Define(baseway, genn(n))
				//
				//left head
				<< Define(lCar, Lambda(zx, If(Length(zx)>0, zx[0], V())))
				//left tail
				<< Define(ccopy, Lambda(tmp, j, ans, If(j>=Length(tmp)-1,
														ans,
														ccopy(tmp, j+1, Cons(ans, tmp[j]))
														)))
				<< Define(lCdr, Lambda(cx, ccopy(cx, V(0), List())))
				//функци€ соединени€ двух списков
				<< Define(merge, Lambda(xs, ys, If(Nullp(ys),
													xs,
													merge(Cons(xs, lCar(ys)), lCdr(ys))
													)))
				//функци€ удалени€ i-го элемента из списка (первый индекс 1, а не 0)
				<< Define(delinlist, Lambda(head, lst, i, If(i<=1, 
															merge(head, lCdr(lst)),
															delinlist(Cons(head, lCar(lst)), lCdr(lst), i-1)
															)))
				<< Define(delRef, Lambda(vs, k, delinlist(List(), vs, k)))
				//1. перестановка методом ƒуршенфельда
				//не работает предположительно из-за неправильного использовани€ let
				/*<< Define(gennn, Lambda(ssnd, ffst, gg, Block(
														Define(ind, RndInt(Length(ffst))+1), 
														If(gg>0,
															List(ffst, ssnd),
															gennn(Cons(ssnd, ffst[ind-1]), delRef(ffst, ind), gg+1)
															) 
														)))
				<< Define(goalway, gennn(List(), baseway, V(0)))*/
				//2. перестановка методом ƒуршенфельда
				//написано некрасиво, но вроде работает
				<< Define(por, Lambda(ind, If(ind<=0,
											List(),
											Cons(por(ind-1), RndInt(ind)+1)
											)))
				<< Define(porr, por(n))
				<< Define(gennn, Lambda(ssnd, ffst, porrr, If(Length(porrr)<=0,
														ssnd,
														gennn(Cons(ssnd, ffst[porrr[Length(ffst)-1]-1]), delRef(ffst, porrr[Length(ffst)-1]), Cdr(porrr))
														) 
													))
				<< Define(goalway, gennn(List(), baseway, porr))
				//sqr
				<< Define(msqr, Lambda(x, x*x))
				//длина отрезка
				<< Define(plength, Lambda(fst, snd,
										Sqrt(msqr(ListRef(fst, V(0))-ListRef(snd, V(0))) + msqr(ListRef(fst, V(1))-ListRef(snd, V(1)))) ))
				//длина всего пути
				<< Define(fulllen, Lambda(pp, If(goalway[pp]-1<=0,
									V(0),
									plength(ListRef(pts, pp), ListRef(pts, goalway[pp]-1))+fulllen(goalway[pp]-1)
									)))
				/*
				<< Define(indOf, Lambda(bs, gint, If(lCar(bs)=gint,
												,
											)
										))
										*/
				<< goalway
				<< fulllen(V(0))
				//<< plength(ListRef(pts, V(0)), ListRef(pts, goalway[0]-1))
		)->reeval(&env, NULL, 1);

#endif

    cout << p->getValue() << endl;


	/*
    cout <<
    (::GaussianBlur(DrawCircle(V(Mat::zeros(img.rows, img.cols, CV_8UC1)),
                              V(50), V(50), V(10), V(255)),
                   V(13), V(0)) * 5).eval()->getValue();
				   */
	int tmpzxcv;
    cv::waitKey();
	cin >> tmpzxcv;

    //p = p->reeval(&env);
    //cout << p->getChild(0)->getValue() << endl;


    
    
    return 0;
}


