import cv2
import numpy as np
import argparse


def displayImg(img):
    """
    画像を表示
    """
    cv2.imshow("display image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def trimImg(img):
    """
    画像の空白部分を除去
    """
    if np.sum(img[0]) == 0:
        return trimImg(img[1:])
    if np.sum(img[-1]) == 0:
        return trimImg(img[:-2])
    if np.sum(img[:, 0]) == 0:
        return trimImg(img[:, 1:])
    if np.sum(img[:, -1]) == 0:
        return trimImg(img[:, :-2])
    return img


def main(args):
    """
    stitchingモジュールを使用しない場合
    """
    # 画像読み込み
    img1 = cv2.imread(args.img_path[0])
    img2 = cv2.imread(args.img_path[1])

    # グレースケール化
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT特徴抽出器を生成
    sift = cv2.SIFT_create()

    # 特徴点(keypoints),特徴記述子(descriptors)を求める
    keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

    # 画像表示
    if args.display:
        img_sift1 = cv2.drawKeypoints(
            img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        img_sift2 = cv2.drawKeypoints(
            img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        displayImg(img_sift1)
        displayImg(img_sift2)

    # 総当たりマッチング(BFMatcher型のオブジェクトを生成)
    bf = cv2.BFMatcher()

    # k近傍法によるマッチング:マッチングスコアが上位k個の特徴点を返す
    matches = bf.knnMatch(descriptors2, descriptors1, k=args.k)

    # ratio test
    good = []
    for m, n in matches:
        if m.distance < args.th * n.distance:
            good.append(m)

    # マッチング画像表示
    if args.display:
        img_matching = cv2.drawMatches(
            img2, keypoints2, img1, keypoints1, good, None, flags=2
        )
        displayImg(img_matching)

    # 合成に必要な数の対応点があるか確認
    assert len(good) > args.min_match, "Not enought matches are found- {}/{}".format(
        len(good), args.min_match
    )

    # 2つの画像の対応する特徴点の座標を抽出
    src_pts = np.float32([keypoints2[m.queryIdx].pt for m in good])
    dst_pts = np.float32([keypoints1[m.trainIdx].pt for m in good])

    # ホモグラフィ行列を生成
    H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # ホモグラフィ行列によってimg2を変換
    img2_warped = cv2.warpPerspective(
        img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0])
    )

    # 変換後のimg2画像を表示
    if args.display:
        displayImg(img2_warped)

    # img1と結合
    img_stitched = img2_warped.copy()
    img_stitched[: img1.shape[0], : img1.shape[1]] = img1

    # 合成画像を表示
    if args.display:
        displayImg(img_stitched)

    # 不要部分を除去した画像を表示
    img_stitched_trimmed = trimImg(img_stitched)
    if args.display:
        displayImg(img_stitched_trimmed)

    # 保存
    if args.save:
        cv2.imwrite("./result.jpg", img_stitched_trimmed)


def main2(args):
    """
    stitchingモジュールを使用した場合
    """
    img1 = cv2.imread(args.img_path[0])
    img2 = cv2.imread(args.img_path[1])

    imgs = []
    imgs.append(img1)
    imgs.append(img2)

    stitcher = cv2.Stitcher_create()
    img_stitched = stitcher.stitch(imgs)[1]

    # 合成画像を表示
    if args.display:
        displayImg(img_stitched)

    # 保存
    if args.save:
        cv2.imwrite("./result2.jpg", img_stitched)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image stitching")
    parser.add_argument(
        "--img_path",
        default=["./img/landscape1.jpg", "./img/landscape2.jpg"],
        help="入力画像パス",
    )
    parser.add_argument("--display", action="store_false", help="表示フラグ")
    parser.add_argument("--save", action="store_true", help="保存フラグ")
    parser.add_argument("--k", type=int, default=2, help="k近傍パラメーター")
    parser.add_argument("--th", type=float, default=0.75, help="ratio testの閾値")
    parser.add_argument("--min_match", type=int, default=10, help="合成に必要な数の対応点数")
    args = parser.parse_args()

    assert len(args.img_path) == 2, "length of img_path is not 2. Need length is 2"

    main(args)
    main2(args)
