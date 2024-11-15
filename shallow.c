#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#if defined(_OPENMP)
#include <omp.h>
#define GET_TIME() (omp_get_wtime()) // wall time
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC) // cpu time
#endif

// h is the depth at rest
// eta represente l'élévation de la surface
// U = depth-averaged velocity
//gamma is the dissipation coefficient

struct parameters 
{
  double dx, dy, dt, max_t;
  double g, gamma;
  int source_type;
  int sampling_rate;
  char input_h_filename[256];
  char output_eta_filename[256];
  char output_u_filename[256];
  char output_v_filename[256];
};


struct data {
  int nx, ny; // number of nodes along x and y
  double dx, dy; //  spatial grid step along x, y
  double *values; //values c'est comme le board mais en une dimension pour que ca soit plus efficace 
};

typedef enum neighbor 
{
  UP    = 0,
  DOWN  = 1,
  LEFT  = 2,
  RIGHT = 3

} neighbor_t;

#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))

int read_parameters(struct parameters *param, const char *filename)
{
  FILE *fp = fopen(filename, "r");
  if(!fp) {
    printf("Error: Could not open parameter file '%s'\n", filename);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fscanf(fp, "%lf", &param->dx) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->dy) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->dt) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->max_t) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->g) == 1);
  if(ok) ok = (fscanf(fp, "%lf", &param->gamma) == 1);
  if(ok) ok = (fscanf(fp, "%d", &param->source_type) == 1);
  if(ok) ok = (fscanf(fp, "%d", &param->sampling_rate) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->input_h_filename) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->output_eta_filename) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->output_u_filename) == 1);
  if(ok) ok = (fscanf(fp, "%256s", param->output_v_filename) == 1);
  fclose(fp);
  if(!ok) {
    printf("Error: Could not read one or more parameters in '%s'\n", filename);
    return 1;
  }
  return 0;
}

void print_parameters(const struct parameters *param)
{
  printf("Parameters:\n");
  printf(" - grid spacing (dx, dy): %g m, %g m\n", param->dx, param->dy);
  printf(" - time step (dt): %g s\n", param->dt);
  printf(" - maximum time (max_t): %g s\n", param->max_t);
  printf(" - gravitational acceleration (g): %g m/s^2\n", param->g);
  printf(" - dissipation coefficient (gamma): %g 1/s\n", param->gamma);
  printf(" - source type: %d\n", param->source_type);
  printf(" - sampling rate: %d\n", param->sampling_rate);
  printf(" - input bathymetry (h) file: '%s'\n", param->input_h_filename);
  printf(" - output elevation (eta) file: '%s'\n", param->output_eta_filename);
  printf(" - output velocity (u, v) files: '%s', '%s'\n",
         param->output_u_filename, param->output_v_filename);
}

int read_data(struct data *data, const char *filename)
{
  FILE *fp = fopen(filename, "rb");
  if(!fp) {
    printf("Error: Could not open input data file '%s'\n", filename);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fread(&data->nx, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fread(&data->ny, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fread(&data->dx, sizeof(double), 1, fp) == 1);
  if(ok) ok = (fread(&data->dy, sizeof(double), 1, fp) == 1);
  if(ok) {
    int N = data->nx * data->ny;
    if(N <= 0) {
      printf("Error: Invalid number of data points %d\n", N);
      ok = 0;
    }
    else {
      data->values = (double*)malloc(N * sizeof(double));
      if(!data->values) {
        printf("Error: Could not allocate data (%d doubles)\n", N);
        ok = 0;
      }
      else {
        ok = (fread(data->values, sizeof(double), N, fp) == N);
      }
    }
  }
  fclose(fp);
  if(!ok) {
    printf("Error reading input data file '%s'\n", filename);
    return 1;
  }
  return 0;
}

int write_data(const struct data *data, const char *filename, int step)
{
  char out[512];
  if(step < 0)
    sprintf(out, "%s.dat", filename);
  else
    sprintf(out, "%s_%d.dat", filename, step);
  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output data file '%s'\n", out);
    return 1;
  }
  int ok = 1;
  if(ok) ok = (fwrite(&data->nx, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->ny, sizeof(int), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->dx, sizeof(double), 1, fp) == 1);
  if(ok) ok = (fwrite(&data->dy, sizeof(double), 1, fp) == 1);
  int N = data->nx * data->ny;
  if(ok) ok = (fwrite(data->values, sizeof(double), N, fp) == N);
  fclose(fp);
  if(!ok) {
    printf("Error writing data file '%s'\n", out);
    return 1;
  }
  return 0;
}

int write_data_vtk(const struct data *data, const char *name,
                   const char *filename, int step)
{
  char out[512];
  if(step < 0)
    sprintf(out, "%s.vti", filename);
  else
    sprintf(out, "%s_%d.vti", filename, step);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK file '%s'\n", out);
    return 1;
  }

  unsigned long num_points = data->nx * data->ny;
  unsigned long num_bytes = num_points * sizeof(double);

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" "
          "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
  fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d 0 0\" "
          "Spacing=\"%lf %lf 0.0\">\n",
          data->nx - 1, data->ny - 1, data->dx, data->dy);
  fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 0\">\n",
          data->nx - 1, data->ny - 1);

  fprintf(fp, "      <PointData Scalars=\"scalar_data\">\n");
  fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" "
          "format=\"appended\" offset=\"0\">\n", name);
  fprintf(fp, "        </DataArray>\n");
  fprintf(fp, "      </PointData>\n");

  fprintf(fp, "    </Piece>\n");
  fprintf(fp, "  </ImageData>\n");

  fprintf(fp, "  <AppendedData encoding=\"raw\">\n_");

  fwrite(&num_bytes, sizeof(unsigned long), 1, fp);
  fwrite(data->values, sizeof(double), num_points, fp);

  fprintf(fp, "  </AppendedData>\n");
  fprintf(fp, "</VTKFile>\n");

  fclose(fp);
  return 0;
}

int write_manifest_vtk(const char *name, const char *filename,
                       double dt, int nt, int sampling_rate)
{
  char out[512];
  sprintf(out, "%s.pvd", filename);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK manifest file '%s'\n", out);
    return 1;
  }

  fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" "
          "byte_order=\"LittleEndian\">\n");
  fprintf(fp, "  <Collection>\n");
  for(int n = 0; n < nt; n++) {
    if(sampling_rate && !(n % sampling_rate)) {
      double t = n * dt;
      fprintf(fp, "    <DataSet timestep=\"%g\" file='%s_%d.vti'/>\n", t,
              filename, n);
    }
  }
  fprintf(fp, "  </Collection>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
  return 0;
}

int init_data(struct data *data, int nx, int ny, double dx, double dy,
              double val)
{
  data->nx = nx;
  data->ny = ny;
  data->dx = dx;
  data->dy = dy;
  data->values = (double*)malloc(nx * ny * sizeof(double));
  if(!data->values){
    printf("Error: Could not allocate data\n");
    return 1;
  }
  for(int i = 0; i < nx * ny; i++) data->values[i] = val;
  return 0;
}

void free_data(struct data *data)
{
  free(data->values);
}

//interpolate data doit retourner le H, le H est constant mais au début on doit le trouver
//Donc au début on a la bathymetric map (data) pas précise, puis on fait un truc plus précis 
double interpolate_data(const struct data *data, double x, double y)
{
  // x, y représentent la position du noeud qu'on interpole
  int i = (int)(x / data->dx);
  int j = (int)(y / data->dy);
  int i1;
  int j1;
  if(i < 0) i = 0;
  if(i > data->nx - 1){ 
    i1 = data->nx - 1;
    i = data-> nx - 2;
  }
  else
  {
    i1 = i + 1; 
  }
  if(j < 0) j = 0;
  if(j > data->ny - 1){ 
    j1 = data->ny - 1;
    j = data-> ny - 2;
  }
  else{
    j1 = j + 1;
  }
  double h00 = GET(data, i, j);
  double h01 = GET(data, i, j1); 
  double h10 = GET(data, i1, j);
  double h11 = GET(data, i1, j1);
  return (h00 * (i1 * data->dx - x) * (j1 * data->dx - y) + 
          h01 * (i1 * data->dx - x) * (y - j * data->dx) + 
          h10 * (x - i * data->dx) * (j1 * data->dx - y) + 
          h11 * (x - i * data->dx) * (y - j * data->dx)) /(data->dx * data->dy);
}

int main(int argc, char **argv)
{
  
  if(argc != 2) {
    printf("Usage: %s parameter_file\n", argv[0]);
    return 1;
  }

   //calcule des dimensions adaptés aux différents process

  int world_size;
  int rank, cart_rank;

  int dims[2] = {0, 0};
  int periods[2] ={0, 0};
  int reorder = 0;

  int coords[2];
  // coords[0] est le nombre de ligne 
  // coords[1] est le nombre de colonne

  int neighbors[4];

  MPI_Comm cart_comm;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Dims_create(world_size, 2, dims);

  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
  MPI_Comm_rank(cart_comm, &cart_rank);

  MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

  MPI_Cart_shift(cart_comm, 0, 1, 
                  &neighbors[UP], &neighbors[DOWN]);

  MPI_Cart_shift(cart_comm, 1, 1, 
                  &neighbors[LEFT], &neighbors[RIGHT]);

  printf("Rank = %4d - Coords = (%3d, %3d) - Neighbors (up, down, left, right) = (%3d, %3d, %3d, %3d)\n",
            rank, coords[0], coords[1], 
            neighbors[UP], neighbors[DOWN], neighbors[LEFT], neighbors[RIGHT]);



  struct parameters param;
  if(read_parameters(&param, argv[1])) return 1;
  

  struct data h;
  // printf("1\n");
  if(read_data(&h, param.input_h_filename)) return 1;
  // printf("2\n");

  //hx, hy représente longueurs/largeurs de la grille (du problème en metre)
  double hx = h.nx * h.dx;
  double hy = h.ny * h.dy;

  // le nombre de noeuds que en fonction des arguments de l'exécution
  int nx = floor(hx / param.dx);
  int ny = floor(hy / param.dy);
  
  if(nx <= 0) nx = 1;
  if(ny <= 0) ny = 1;

  // le nombre de pas de temps
  int nt = floor(param.max_t / param.dt);

  if(rank == 0)
  {
    printf(" - grid size: %g m x %g m (%d x %d = %d grid points)\n",
          hx, hy, nx, ny, nx * ny);
    printf(" - number of time steps: %d\n", nt);
    printf("nx : %d, ny: %d\n", nx, ny);
    print_parameters(&param);
  }


  // px = taille selon x du process

  int px, py, startpx, startpy, endpx, endpy; 
  px = (int) nx / dims[0];
  py = (int) ny / dims[1];
  startpx = coords[0] * px;
  startpy = coords[1] * py;
  endpx = coords[0] == dims[0] - 1 ? nx - 1: startpx + px - 1;
  endpy = coords[1] == dims[1] - 1 ? ny - 1: startpy + py - 1;
  px = endpx - startpx + 1;
  py = endpy - startpy + 1;
  // printf("Process %d out of %d, position : %d, %d, interval : [%d, %d]*[%d, %d]\n", rank, world_size, coords[0], coords[1], startpx, endpx, startpy, endpy);
  //tu crées 3 nouvelles structure data
  struct data eta, u, v;
  init_data(&eta, px, py, param.dx, param.dx, 0.);
  init_data(&u, coords[0] == 0 ? px + 1 : px, py, param.dx, param.dy, 0.);
  init_data(&v, px, coords[1] == 0 ? py + 1 : py, param.dx, param.dy, 0.);

  // interpolate bathymetry
  //les h interp c'est les h plus précis que la bathymétric map
  struct data h_interp;
  // h_u et h_v c'est la hauteur aux endroits où on évalue la vitesse
  struct data h_u;
  struct data h_v;
  init_data(&h_interp, px, py, param.dx, param.dy, 0.);
  init_data(&h_u, px + 1, py, param.dx, param.dy, 0.);
  init_data(&h_v, px, py + 1, param.dx, param.dy, 0.);


 

  double start = GET_TIME();



  
  for(int i = 0; i < px ; i++) 
  {
    for(int j = 0; j < py; j++) 
    {
      double x = i * param.dx;
      double y = j * param.dy;
      //ici on interpole à partir de h (donc la bathymétric map pas précise)
      double val = interpolate_data(&h, startpx + x, startpy + y);
      SET(&h_interp, i, j, val);
      val = interpolate_data(&h, startpx + x + param.dx / 2, startpy + y);
      SET(&h_u, i, j, val);
      val = interpolate_data(&h, startpx + x, startpy + y + param.dy / 2);
      SET(&h_v, i, j, val); 
    }
  }

  MPI_Request request1, request2, request3, request4;
  int num_requests = 0;

  if(coords[0] != 0)
  {
    // Appeler la méthode qui le reçoit


    double* left_col_hu; 
    
    MPI_Irecv(left_col_hu, py, MPI_DOUBLE, neighbors[LEFT], neighbors[LEFT], cart_comm, &request1);
    
  }
  if(coords[0] != dims[0] - 1)
  {
    //Send the last col of hu to NEIGHBORS[RIGHT]

    double *right_col_hu = (double *)malloc(py * sizeof(double)); //au lieu de faire une copie, essayé de directement encoyé l'adresse de la colonne

    for(int i = 0; i < py; i++)
    {
      right_col_hu[i] = GET(&h_u, i, px -1);
    }

    
    MPI_Isend(right_col_hu, py, MPI_DOUBLE, neighbors[RIGHT], rank, cart_comm, &request2);
    
  }
  if(coords[1] != 0)
  {
  
    double* up_row_hv; 
    MPI_Irecv(up_row_hv, px, MPI_DOUBLE, neighbors[UP], neighbors[UP], cart_comm, &request3);

  }
  if(coords[1] != dims[1] - 1)
  {
    //Send the last row of hv to NEIGHBOR[LEFT]

    double *down_row_hv = (double *)malloc(px * sizeof(double));
    for(int j = 0; j < px; j++)
    {
      down_row_hv[j] = GET(&h_v, py - 1,  j);
    }

    MPI_Isend(down_row_hv, px, MPI_DOUBLE, neighbors[DOWN], rank, cart_comm, &request4);

  }

  MPI_Wait(&request1, MPI_STATUS_IGNORE);
  MPI_Wait(&request2, MPI_STATUS_IGNORE);
  MPI_Wait(&request3, MPI_STATUS_IGNORE);
  MPI_Wait(&request4, MPI_STATUS_IGNORE);
  
  

  // boucle temporelle
  for(int n = 0; n < nt; n++) 
  {

    if(n && (n % (nt / 10)) == 0) 
    {
      double time_sofar = GET_TIME() - start;
      double eta = (nt - n) * time_sofar / n;
      printf("Computing step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
      fflush(stdout);
    }

    // output solution
    if(param.sampling_rate && !(n % param.sampling_rate)) {
      write_data_vtk(&eta, "water elevation", param.output_eta_filename, n);
      //write_data_vtk(&u, "x velocity", param.output_u_filename, n);
      //write_data_vtk(&v, "y velocity", param.output_v_filename, n);
    }

    // impose boundary conditions
    double t = n * param.dt;
    if(param.source_type == 1) {
      // sinusoidal velocity on top boundary
      double A = 5;
      double f = 1. / 20.;
      for(int i = 0; i < px; i++) 
      {
        for(int j = 0; j < py; j++) 
        {
          SET(&u, 0, j, 0.);
          SET(&u, px, j, 0.);
          SET(&v, i, 0, 0.);
          SET(&v, i, py, A * sin(2 * M_PI * f * t));
        }
      }
    }
    else if(param.source_type == 2) 
    {
      // sinusoidal elevation in the middle of the domain
      double A = 5;
      double f = 1. / 20.;
      SET(&eta, px / 2, py / 2, A * sin(2 * M_PI * f * t));
    }
    else 
    {
      // TODO: add other sources
      printf("Error: Unknown source type %d\n", param.source_type);
      exit(0);
    }

    // update eta


    MPI_Request request5, request6, request7, request8;
    
    if(coords[0] != 0)
    {
      double* left_col_u; 
      MPI_Irecv(left_col_u, u.ny, MPI_DOUBLE, neighbors[LEFT], neighbors[LEFT], cart_comm, &request5);
      // Appeler la méthode qui le reçoit
    }
    if(coords[0] != dims[0] - 1)
    {
      //Send the last col of u

      double *right_col_u = (double *)malloc(u.ny * sizeof(double));
      for(int i = 0; i < u.ny; i++)
      {
        right_col_u[i] = GET(&u, i, u.nx -1);
      }

      MPI_Isend(right_col_u, u.ny, MPI_DOUBLE, neighbors[RIGHT], rank, cart_comm, &request6);


    }
    if(coords[1] != 0)
    {
      double* up_row_v; 
      MPI_Irecv(up_row_v, v.nx, MPI_DOUBLE, neighbors[UP], neighbors[UP], cart_comm, &request7);
    }
    if(coords[1] != dims[1] - 1)
    {
      //Send the last row of v
      double *down_row_v = (double *)malloc(v.nx * sizeof(double));

      for(int i = 0; i < v.nx; i++)
      {
        down_row_v[i] = GET(&v, v.ny -1, i);
      }

      MPI_Isend(down_row_v, v.nx, MPI_DOUBLE, neighbors[DOWN], rank, cart_comm, &request8);
    }

    for(int i = 1; i < px; i++) 
    {
      for(int j = 1; j < py; j++) 
      {
        //If we are in one border, there is one more 
        double hui1j = coords[0] == 0 ? GET(&h_u, i + 1, j)  : GET(&h_u, i, j);
        double huij = coords[0] == 0 ? GET(&h_u, i, j)  : GET(&h_u, i-1, j);
        double hvij1 = coords[1] == 0 ? GET(&h_v, i, j - 1)  : GET(&h_v, i, j);
        double hvij = coords[1] == 0 ? GET(&h_v, i, j)  : GET(&h_v, i, j-1);
        
        
        double eta_ij = GET(&eta, i, j)
          - param.dt / param.dx * (hui1j * GET(&u, i + 1, j) - huij * GET(&u, i, j))
          - param.dt / param.dy * (hvij1 * GET(&v, i, j + 1) - hvij * GET(&v, i, j));
        SET(&eta, i, j, eta_ij);
      }
    }
    // Verifier que tout a bien été reçu

    
    MPI_Wait(&request5, MPI_STATUS_IGNORE);
    MPI_Wait(&request6, MPI_STATUS_IGNORE);
    MPI_Wait(&request7, MPI_STATUS_IGNORE);
    MPI_Wait(&request8, MPI_STATUS_IGNORE);

    for(int i = 0; i < px; i++){
      double hui1j = i == 0 && coords[0] == 0 ? GET(&h_u, 0, 0) : GET(&h_u, );//REFLECHIR
      double huij = coords[0] == 0 ? GET(&h_u, i, 0) : left_col_hu[i];
      double hvij1 = coords[1] == 0 ? GET(&h_v, i, 1) : GET(&h_v, i, 0);
      double hvij = coords[1] == 0 ? GET(&h_u, i, 0) : up_col_hv[0];
      
      double eta_ij = GET(&eta, i, 0)
        - param.dt / param.dx * (hui1j * GET(&u, i + 1, j) - huij * GET(&u, i, j))
        - param.dt / param.dy * (hvij1 * GET(&v, i, j + 1) - hvij * GET(&v, i, j));
      SET(&eta, i, 0, eta_ij);
    }
    if(coords[1] != 0){
      for(int i = 0; i < px; i++){
        double hui1j = GET(&h_u, 0, i);
        double huij = i == 0 ? left_col_hu[0] : GET(&h_u, i, j);
        double hvij1 = GET(&h_v, 0, i + 1);
        double hvij = up_col_hv[0]
        
        double eta_ij = GET(&eta, i, 0)
          - param.dt / param.dx * (hui1j * GET(&u, i + 1, j) - huij * GET(&u, i, j))
          - param.dt / param.dy * (hvij1 * GET(&v, i, j + 1) - hvij * GET(&v, i, j));
        SET(&eta, 0, i, eta_ij);
      }
    }



    // update u and v

    MPI_Request request9, request10, request11, request12;
    
    if(coords[0] != 0)
    {
      double* left_col_eta; 
      MPI_Irecv(left_col_eta, eta.ny, MPI_DOUBLE, neighbors[LEFT], neighbors[LEFT], cart_comm, &request9);
      // Appeler la méthode qui le reçoit
    }
    if(coords[0] != dims[0] - 1)
    {
      //Send the last col of u

      double *right_col_eta = (double *)malloc(eta.ny * sizeof(double));
      for(int i = 0; i < eta.ny; i++)
      {
        right_col_eta[i] = GET(&eta, i, eta.nx -1);
      }

      MPI_Isend(right_col_eta, eta.ny, MPI_DOUBLE, neighbors[RIGHT], rank, cart_comm, &request10);


    }
    if(coords[1] != 0)
    {
      double* up_row_eta; 
      MPI_Irecv(up_row_eta, eta.nx, MPI_DOUBLE, neighbors[UP], neighbors[UP], cart_comm, &request11);
    }
    if(coords[1] != dims[1] - 1)
    {
      //Send the last row of v
      double *down_row_eta = (double *)malloc(eta.nx * sizeof(double));

      for(int i = 0; i < eta.nx; i++)
      {
        down_row_eta[i] = GET(&v, eta.ny -1, i);
      }

      MPI_Isend(down_row_eta, eta.nx, MPI_DOUBLE, neighbors[DOWN], rank, cart_comm, &request12);
    }

    // il va falloir séparer en 2 pour le cas avec u et v puisqu'ils ne font pas la même taille. Ce sera plus simple

    for(int i = 0; i < nx; i++) 
    {
      for(int j = 0; j < ny; j++) 
      {
        double c1 = param.dt * param.g;
        double c2 = param.dt * param.gamma;
        double eta_ij = GET(&eta, i, j);
        double eta_imj = GET(&eta, (i == 0) ? 0 : i - 1, j);
        double eta_ijm = GET(&eta, i, (j == 0) ? 0 : j - 1);
        double u_ij = (1. - c2) * GET(&u, i, j)
          - c1 / param.dx * (eta_ij - eta_imj);
        double v_ij = (1. - c2) * GET(&v, i, j)
          - c1 / param.dy * (eta_ij - eta_ijm);
        SET(&u, i, j, u_ij);
        SET(&v, i, j, v_ij);
      }
    }

    MPI_Wait(&request9, MPI_STATUS_IGNORE);
    MPI_Wait(&request10, MPI_STATUS_IGNORE);
    MPI_Wait(&request11, MPI_STATUS_IGNORE);
    MPI_Wait(&request12, MPI_STATUS_IGNORE);

  }

  MPI_Finalize();

  write_manifest_vtk("water elevation", param.output_eta_filename,
                     param.dt, nt, param.sampling_rate);
  //write_manifest_vtk("x velocity", param.output_u_filename,
  //                   param.dt, nt, param.sampling_rate);
  //write_manifest_vtk("y velocity", param.output_v_filename,
  //                   param.dt, nt, param.sampling_rate);

  double time = GET_TIME() - start;
  printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
         1e-6 * (double)eta.nx * (double)eta.ny * (double)nt / time);

  // Il faut free les h_u ...
  free_data(&h_interp);
  free_data(&eta);
  free_data(&u);
  free_data(&v);

  return 0;
}