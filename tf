%!PS-Adobe-3.0
%%Creator: (ImageMagick)
%%Title: (tf)
%%CreationDate: (2018-09-10T00:58:43+08:00)
%%%%BoundingBox: (atend)
%%DocumentData: Clean7Bit
%%LanguageLevel: 1
%%Orientation: Portrait
%%PageOrder: Ascend
%%Pages: 3
%%EndComments

%%BeginDefaults
%%EndDefaults

%%BeginProlog
%
% Display a color image.  The image is displayed in color on
% Postscript viewers or printers that support color, otherwise
% it is displayed as grayscale.
%
/DirectClassPacket
{
  %
  % Get a DirectClass packet.
  %
  % Parameters:
  %   red.
  %   green.
  %   blue.
  %   length: number of pixels minus one of this color (optional).
  %
  currentfile color_packet readhexstring pop pop
  compression 0 eq
  {
    /number_pixels 3 def
  }
  {
    currentfile byte readhexstring pop 0 get
    /number_pixels exch 1 add 3 mul def
  } ifelse
  0 3 number_pixels 1 sub
  {
    pixels exch color_packet putinterval
  } for
  pixels 0 number_pixels getinterval
} bind def

/DirectClassImage
{
  %
  % Display a DirectClass image.
  %
  systemdict /colorimage known
  {
    columns rows 8
    [
      columns 0 0
      rows neg 0 rows
    ]
    { DirectClassPacket } false 3 colorimage
  }
  {
    %
    % No colorimage operator;  convert to grayscale.
    %
    columns rows 8
    [
      columns 0 0
      rows neg 0 rows
    ]
    { GrayDirectClassPacket } image
  } ifelse
} bind def

/GrayDirectClassPacket
{
  %
  % Get a DirectClass packet;  convert to grayscale.
  %
  % Parameters:
  %   red
  %   green
  %   blue
  %   length: number of pixels minus one of this color (optional).
  %
  currentfile color_packet readhexstring pop pop
  color_packet 0 get 0.299 mul
  color_packet 1 get 0.587 mul add
  color_packet 2 get 0.114 mul add
  cvi
  /gray_packet exch def
  compression 0 eq
  {
    /number_pixels 1 def
  }
  {
    currentfile byte readhexstring pop 0 get
    /number_pixels exch 1 add def
  } ifelse
  0 1 number_pixels 1 sub
  {
    pixels exch gray_packet put
  } for
  pixels 0 number_pixels getinterval
} bind def

/GrayPseudoClassPacket
{
  %
  % Get a PseudoClass packet;  convert to grayscale.
  %
  % Parameters:
  %   index: index into the colormap.
  %   length: number of pixels minus one of this color (optional).
  %
  currentfile byte readhexstring pop 0 get
  /offset exch 3 mul def
  /color_packet colormap offset 3 getinterval def
  color_packet 0 get 0.299 mul
  color_packet 1 get 0.587 mul add
  color_packet 2 get 0.114 mul add
  cvi
  /gray_packet exch def
  compression 0 eq
  {
    /number_pixels 1 def
  }
  {
    currentfile byte readhexstring pop 0 get
    /number_pixels exch 1 add def
  } ifelse
  0 1 number_pixels 1 sub
  {
    pixels exch gray_packet put
  } for
  pixels 0 number_pixels getinterval
} bind def

/PseudoClassPacket
{
  %
  % Get a PseudoClass packet.
  %
  % Parameters:
  %   index: index into the colormap.
  %   length: number of pixels minus one of this color (optional).
  %
  currentfile byte readhexstring pop 0 get
  /offset exch 3 mul def
  /color_packet colormap offset 3 getinterval def
  compression 0 eq
  {
    /number_pixels 3 def
  }
  {
    currentfile byte readhexstring pop 0 get
    /number_pixels exch 1 add 3 mul def
  } ifelse
  0 3 number_pixels 1 sub
  {
    pixels exch color_packet putinterval
  } for
  pixels 0 number_pixels getinterval
} bind def

/PseudoClassImage
{
  %
  % Display a PseudoClass image.
  %
  % Parameters:
  %   class: 0-PseudoClass or 1-Grayscale.
  %
  currentfile buffer readline pop
  token pop /class exch def pop
  class 0 gt
  {
    currentfile buffer readline pop
    token pop /depth exch def pop
    /grays columns 8 add depth sub depth mul 8 idiv string def
    columns rows depth
    [
      columns 0 0
      rows neg 0 rows
    ]
    { currentfile grays readhexstring pop } image
  }
  {
    %
    % Parameters:
    %   colors: number of colors in the colormap.
    %   colormap: red, green, blue color packets.
    %
    currentfile buffer readline pop
    token pop /colors exch def pop
    /colors colors 3 mul def
    /colormap colors string def
    currentfile colormap readhexstring pop pop
    systemdict /colorimage known
    {
      columns rows 8
      [
        columns 0 0
        rows neg 0 rows
      ]
      { PseudoClassPacket } false 3 colorimage
    }
    {
      %
      % No colorimage operator;  convert to grayscale.
      %
      columns rows 8
      [
        columns 0 0
        rows neg 0 rows
      ]
      { GrayPseudoClassPacket } image
    } ifelse
  } ifelse
} bind def

/DisplayImage
{
  %
  % Display a DirectClass or PseudoClass image.
  %
  % Parameters:
  %   x & y translation.
  %   x & y scale.
  %   label pointsize.
  %   image label.
  %   image columns & rows.
  %   class: 0-DirectClass or 1-PseudoClass.
  %   compression: 0-none or 1-RunlengthEncoded.
  %   hex color packets.
  %
  gsave
  /buffer 512 string def
  /byte 1 string def
  /color_packet 3 string def
  /pixels 768 string def

  currentfile buffer readline pop
  token pop /x exch def
  token pop /y exch def pop
  x y translate
  currentfile buffer readline pop
  token pop /x exch def
  token pop /y exch def pop
  currentfile buffer readline pop
  token pop /pointsize exch def pop
  /Times-Roman findfont pointsize scalefont setfont
  x y scale
  currentfile buffer readline pop
  token pop /columns exch def
  token pop /rows exch def pop
  currentfile buffer readline pop
  token pop /class exch def pop
  currentfile buffer readline pop
  token pop /compression exch def pop
  class 0 gt { PseudoClassImage } { DirectClassImage } ifelse
  grestore
  showpage
} bind def
%%EndProlog
%%Page:  1 1
%%PageBoundingBox: 1028 274 1046 287
DisplayImage
1028 274
18 13
12
18 13
0
0
423630BC7157E86D43E76438E76438E76539D769423A39364E4C454E4C453A39357A7973797872
787771787771787771787771787771985F49E7683DE55E30E55E30E55E30E55E30E3663B383735
4C4A444C4A443938357B7A7375746D75746D75746D3D3C373D3C373D3C37423730B6512DE45C2D
E45C2DE45C2DE45D2ED460373A39354B49434B49433A393671706B72716B71706A71706A7F7F79
7F7F797F7F79BC6B4F42362FBE502BE3592AE3592AE35A2CB153333D3B384A48424A48423C3C37
63625D6F6E686E6D676E6D676E6D676E6D676E6D67E46235C36D50DB5D33E25727E25727E25B2B
74443341403C49474249474242413C4B4B476D6C666A69636A69636A69636A69636A6963E15424
E35D2FE15627E15424E15526C0512D3B39364745414846414846414846403B3A3661605A686761
676660676660676660676660E05221E05221E05221E05322D353274D3A3142403C474540474540
47454047454042413C3E3D3962625C64635E63625D63625D63625DDF501EDF511FDF511FBD4D25
4E3A313F3F3946453F46453F46453F46453F46453F46453F403E3A3D3C39595854605F5B605F5B
5F5E5ACD4C1EAA4623713E2B3B3935403F3B45433E45433E45433E45433E45433E45433E45433E
45433E413F3B3A393645434051504C5A59543938353B3A373E3D3943413C44423D44423D44423D
44423D44423D44423D44423D44423D44423D44423D44423C3E3D393B3A3639383543413D43413D
43413D43413D43413D43413D43413D43413D43413D43413D43413D43413D43413D43413D43413D
43413D43413D43413D42403C42403C42403C42403C42403C42403C42403C42403C42403C42403C
42403C42403C42403C42403C42403C42403C42403C42403C41403B41403B41403B41403B41403B
41403B41403B41403B41403B41403B41403B41403B41403B41403B41403B41403B41403B41403B

%%PageTrailer
%%Page:  1 2
%%PageBoundingBox: 1263 380 1271 385
DisplayImage
1263 380
8 5
12
8 5
0
0
300A24300A24300A24300A24322477CEFDFFF1B98F795161300A24300A24300A24300A24300A2C
5169ADD6EBFCFFFFFF300A24300A24300A24300A24300A24300A24300A24300A24300A24300A24
300A24300A24300A24300A24300A24300A24300A24300A24300A24300A24300A24300A24300A24
300A24

%%PageTrailer
%%Page:  1 3
%%PageBoundingBox: 1300 394 1357 400
DisplayImage
1300 394
57 6
12
57 6
0
0
300A24C5B54A0B0000000000000000000000000000000000003488BBF3DBCFF5DAC6C97D3E1700
0001162E566980BAC3C9F4DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DB
CFF5DBCFF5DBCFF5DBCFF5DBC5B64B0B00226DD0D6CFF5DBCFF5DBCFF5DBCFF5DBC4B4480A001C
62C8D5CFF5DBCFF5DBCFF5CC8D3F02002A784D0F24300A24300A24300A24300A24300A24300A24
300A24300A24300A2430124EA1E2FFCFF5DB3B4BB7F5FFE9AC584D523248523248523248523248
502932330A24300A24300A2431113A5D5F8BABBBDFF8FFFFEEB882541425300A24300A24300A24
300A24300A24300A24300A24300A24300A24300A24300A24300A243A49B4F4FFDD922F29300A24
300A24300A24300A243B4BB7F5FFE39E372A300A24300A24300A3372C0FD002A784D0F24300A24
300A24300A24300A24300A24300A24300A24300A24300A2430124D9EE0FFCFF5DB33297FD5FFFF
DF844B370A24300A24300A24300A24300A24300A24300A24300A24300A24300A24300A263E47A0
E7FFFAC55533300A24300A24300A24300A24300A24300A24300A24300A24300A24300A24300A24
300A243943AAEFFFE6A23A2B300A24300A24300A24300A24332981D6FFFFD77642340A24300A24
31195FB3EDFF0F5CA63A0A24300A24300A24300A24300A24300A24300A24300A24300A24300A24
300D3F86D0FFCFF5DB300C33689DE7FFFFF1CE9A867A54616A546F847A7D5C1825300A24300B32
658AA9A0746F6944525B415E7A8ACAF6FFEFB0452D300A24300A24300A24300A24300A24300A24
300A24300A24300A24300A24300A24300A24322477CEFDFFF1B98F7951616D5E7D86503C340A24
300C356CA4EBFFFDDDB2776E6E577A9FC0F1FFFBD071CFD6300A24300A24300A24300A24300A24
300A24300A24300A24300A24300A24300A294F7AD6CFF5DB300A24300F39677CB2D2E4F7FFFFFF
FFFFF8ECD0AC762427300A24300D396C8DC0DDECFCFFFFFFFFFFFFF6E3D0B578573F0C24300A24
300A24300A24300A24300A24300A24300A24300A24300A24300A24300A24300A24300A2C5169AD
D6EBFCFFFFFFFFF5E3C57B4B360A24300A243012427591C5E3F5FFFFFFFCEDCEB18C4233CDF5DB
300A24300A24300A24300A24300A24300A24300A24300A24300A24300A24300A24301140CFF5DB
CFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DB
CFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DB
CFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DBCFF5DB
CFF5DBCFF5DBCFF5DBCFF5DBCFF5DB300A24300A24300A24300A24300A24300A24300A24300A24
300A24300A24300A24300A24

%%PageTrailer
%%Trailer
%%BoundingBox: 1028 274 1356 399
%%HiResBoundingBox: 1028 274 1356 399
%%EOF
