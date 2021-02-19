

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glxew.h>
#include <Cg/cg.h>
#include <Cg/cgGL.h>
#include <omp.h>




#define NUM_PARTICLES 500000





static CGcontext   myCgContext;
static CGprofile   myCgVertexProfile,myCgFragmentProfile;
static CGprogram   myCgVertexProgram,myCgFragmentProgram;

static CGparameter myCgVertexParam_globalTime;
static CGparameter myCgVertexParam_acceleration;
static CGparameter myCgVertexParam_modelViewProj;


static const char *myVertexProgramFileName   = "C6E2v_particle.cg";
static const char *myVertexProgramName       = "C6E2v_particle";

static const char *myFragmentProgramFileName = "C6E2v_particle.cg";
static const char *myFragmentProgramName     = "texcoord2color";

static void display(void);
static void visibility(int state);
static void keyboard(unsigned char c, int x, int y);
static void menu(int item);
static void requestSynchronizedSwapBuffers(void);
static void resetParticles(void);

int main(int argc, char **argv)
{

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(980, 760);
    glutInitWindowPosition(150,100);

    glutCreateWindow("Particle Simulation");
    glutDisplayFunc(display);
    glutVisibilityFunc(visibility);
    glutKeyboardFunc(keyboard);

    glewInit();
    resetParticles();
    requestSynchronizedSwapBuffers();
    glClearColor(0.0, 0.335, 0.385, 1.0);
    glPointSize(4.0f);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);

    myCgContext = cgCreateContext();
    cgGLSetDebugMode(CG_FALSE);

    myCgVertexProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
    cgGLSetOptimalOptions(myCgVertexProfile);

    myCgVertexProgram =  cgCreateProgramFromFile(  myCgContext,  CG_SOURCE,  myVertexProgramFileName,
                myCgVertexProfile,  myVertexProgramName,  NULL);
    cgGLLoadProgram(myCgVertexProgram);

    myCgVertexParam_globalTime = cgGetNamedParameter(myCgVertexProgram, "globalTime");
    myCgVertexParam_acceleration = cgGetNamedParameter(myCgVertexProgram, "acceleration");
    myCgVertexParam_modelViewProj = cgGetNamedParameter(myCgVertexProgram, "modelViewProj");

    myCgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
    cgGLSetOptimalOptions(myCgFragmentProfile);

    myCgFragmentProgram =  cgCreateProgramFromFile( myCgContext,   CG_SOURCE,
                myFragmentProgramFileName,  myCgFragmentProfile, myFragmentProgramName, NULL);
    cgGLLoadProgram(myCgFragmentProgram);

    glutCreateMenu(menu);
    glutAddMenuEntry("[ ] Animate", ' ');
    glutAddMenuEntry("[p] Toggle point size computation", 'p');
    glutAddMenuEntry("[r] Reset particles", 'r');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    glutMainLoop();
    return 0;
}

static int myAnimating = 1;
static int myVerbose = 0;
static float myGlobalTime = 0.0;
static int myPass = 0;


typedef struct Particle_t {
    float pInitial[3];
    float vInitial[3];
    float tInitial;
    int alive;
} Particle;


static Particle myParticleSystem[NUM_PARTICLES];

static float float_rand(void) { return rand() / (float) RAND_MAX; }
#define RANDOM_RANGE(lo, hi) ((lo) + (hi - lo) * float_rand())

static void resetParticles(void)
{
    int i;
    myGlobalTime = 0.0;
    myPass = 0;
#pragma omp parallel for
    for(i = 0; i<NUM_PARTICLES; i++) {
        float radius = 0.05f;
        float initialElevation = -0.5f;
        myParticleSystem[i].pInitial[0] = radius * cos(i * 0.5f);
        myParticleSystem[i].pInitial[1] = initialElevation;
        myParticleSystem[i].pInitial[2] = radius * sin(i * 0.5f);
        myParticleSystem[i].alive = 0;
        myParticleSystem[i].tInitial = RANDOM_RANGE(0,10);
    }
}

static void advanceParticles(void)
{
    float death_time = myGlobalTime - 1.0;
    int i;

    myPass++;
#pragma omp parallel for
    for(i=0; i<NUM_PARTICLES; i++) {
        if (!myParticleSystem[i].alive &&(myParticleSystem[i].tInitial <= myGlobalTime))
        {
            myParticleSystem[i].vInitial[0] = RANDOM_RANGE(-1,1);
            myParticleSystem[i].vInitial[1] = RANDOM_RANGE(0,8);
            myParticleSystem[i].vInitial[2] = RANDOM_RANGE(-0.5,0.5);
            myParticleSystem[i].tInitial = myGlobalTime;
            myParticleSystem[i].alive = 1;
            if (myVerbose) {
                printf("Birth %d (%f,%f,%f) at %f\n", i,
                       myParticleSystem[i].vInitial[0],
                        myParticleSystem[i].vInitial[1],
                        myParticleSystem[i].vInitial[2], myGlobalTime);
            }
        }
        if (myParticleSystem[i].alive
                && (myParticleSystem[i].tInitial <= death_time)) {
            myParticleSystem[i].alive = 0;
            myParticleSystem[i].tInitial = myGlobalTime + .01; /* Rebirth next pass */
            if (myVerbose) {
                printf("Death %d at %f\n", i, myGlobalTime);
            }
        }
    }
}

static void display(void)
{
    const float acceleration = -9.8;
    const float viewAngle = myGlobalTime * 2.8;
    int i;
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    gluLookAt(cos(viewAngle), 0.3, sin(viewAngle),   0, 0, 0,   0, 1, 0);

    cgSetParameter1f(myCgVertexParam_globalTime, myGlobalTime);
    cgSetParameter4f(myCgVertexParam_acceleration, 0, acceleration, 0, 0);
    cgGLSetStateMatrixParameter(myCgVertexParam_modelViewProj,
                                CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY);

    cgGLBindProgram(myCgVertexProgram);
    cgGLEnableProfile(myCgVertexProfile);
    cgGLBindProgram(myCgFragmentProgram);
    cgGLEnableProfile(myCgFragmentProfile);

    if (myVerbose) {
        printf("Pass %d\n", myPass);
    }

    glBegin(GL_POINTS);
#pragma omp parallel for
    for(i=0; i<NUM_PARTICLES; i++) {
        if (myParticleSystem[i].alive) {
            glTexCoord3fv(myParticleSystem[i].vInitial);
            glMultiTexCoord1f(GL_TEXTURE1, myParticleSystem[i].tInitial);
            glVertex3fv(myParticleSystem[i].pInitial);
            if (myVerbose) {
                printf("Drew %d (%f,%f,%f) at %f\n", i,
                       myParticleSystem[i].vInitial[0],
                        myParticleSystem[i].vInitial[1],
                        myParticleSystem[i].vInitial[2], myGlobalTime);
            }
        }
    }
    glEnd();
    cgGLDisableProfile(myCgVertexProfile);
    cgGLDisableProfile(myCgFragmentProfile);
    glutSwapBuffers();
}

static void idle(void)
{
    if (myAnimating) {
        myGlobalTime += 0.005;
        advanceParticles();
    }
    glutPostRedisplay();
}

static void visibility(int state)
{
    if (state == GLUT_VISIBLE && myAnimating) {
        glutIdleFunc(idle);
    } else {
        glutIdleFunc(NULL);
    }
}

static void keyboard(unsigned char c, int x, int y)
{
    static int useComputedPointSize = 0;
    if(x==y) x=y;
    switch (c) {
    case 27:
        cgDestroyProgram(myCgVertexProgram);
        cgDestroyProgram(myCgFragmentProgram);
        cgDestroyContext(myCgContext);
        exit(0);
        break;
    case ' ':
        myAnimating = !myAnimating;
        if (myAnimating) {
            glutIdleFunc(idle);
        } else {
            glutIdleFunc(NULL);
        }
        break;
    case 'p':
        useComputedPointSize = !useComputedPointSize;
        if (useComputedPointSize) {
            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
        } else {
            glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
        }
        glutPostRedisplay();
        break;
    case 'r':
        resetParticles();
        glutPostRedisplay();
        break;
    case 'v':
        myVerbose = !myVerbose;
        glutPostRedisplay();
    }
}

static void menu(int item)
{
    keyboard((unsigned char)item, 0, 0);
}


static void requestSynchronizedSwapBuffers(void)
{
    if (glXSwapIntervalSGI) {
        glXSwapIntervalSGI(1);
    }

}
